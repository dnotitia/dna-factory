import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import datasets
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "build_frontier_grpo_dataset", ROOT / "utils" / "build-frontier-grpo-dataset.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

GOOD = (
    "<solution>\n<step>Six times seven is 42.</step>\n"
    "<step>So the product is 42.</step>\n<answer>42</answer>\n</solution>"
)


def test_teacher_output_is_parsed_into_steps_and_one_answer():
    steps, answer = builder.parse_teacher_output(GOOD)
    assert steps == ["Six times seven is 42.", "So the product is 42."]
    assert answer == "42"


@pytest.mark.parametrize(
    "text",
    [
        "The answer is 42.",  # no tags
        "<solution><step>a</step></solution>",  # no answer
        "<step>a</step><answer>1</answer><answer>2</answer>",  # two answers
        "<solution><answer>42</answer></solution>",  # no steps
        None,
    ],
)
def test_teacher_output_off_format_is_rejected(text):
    assert builder.parse_teacher_output(text) == ([], None)


def test_teacher_answer_is_checked_like_the_accuracy_reward():
    assert builder.answer_verifies("42", "42")
    assert builder.answer_verifies("\\frac{1}{2}", "0.5")  # equivalent, not identical
    assert not builder.answer_verifies("41", "42")


def test_teach_keeps_only_a_verified_trace_and_retries():
    replies = iter(["no tags at all", GOOD])
    assert builder.teach("6*7?", "42", lambda q: next(replies), attempts=2) == [
        "Six times seven is 42.",
        "So the product is 42.",
    ]
    # Right format, wrong answer: a trace that would teach the wrong thing is dropped.
    assert builder.teach("6*7?", "43", lambda q: GOOD, attempts=2) == []

    def broken(question):
        raise ConnectionError("teacher down")

    assert builder.teach("6*7?", "42", broken, attempts=2) == []


def test_row_has_the_columns_the_curriculum_reads():
    row = builder.build_row("What is 6*7?", "42", ["a", " b "], "SYS", "Box it.")
    assert row == {
        "prompt": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "What is 6*7?\n\nBox it."},
        ],
        "solution": "\\boxed{42}",
        "steps": ["a", "b"],
    }
    assert builder.build_row("q", "\\boxed{1}", None, "", "")["prompt"] == [
        {"role": "user", "content": "q"}
    ]


def test_early_answer_leak_is_detected_but_not_in_the_last_step():
    assert builder.leaks_early(["so x = 1234", "done"], "1234")
    assert not builder.leaks_early(["setup", "so x = 1234"], "1234")
    assert not builder.leaks_early(
        ["x = 2 here", "done"], "2"
    )  # too short to mean anything


class FakeTeacher(BaseHTTPRequestHandler):
    """OpenAI-compatible endpoint: right for the product, wrong for the sum, garbage otherwise."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        question = body["messages"][-1]["content"]
        assert body["messages"][0]["content"] == builder.TEACHER_SYSTEM
        assert self.headers["Authorization"] == "Bearer test-key"
        if "6*7" in question:
            content = GOOD
        elif "1+1" in question:
            content = "<solution><step>1+1 is 3.</step><answer>3</answer></solution>"
        else:
            content = "I cannot solve this."
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_builder_generates_verifies_and_writes_a_dataset(tmp_path, monkeypatch, capsys):
    source = datasets.Dataset.from_list(
        [
            {"q": "What is 6*7?", "a": "42"},
            {"q": "What is 1+1?", "a": "2"},
            {"q": "Prove the Riemann hypothesis.", "a": "true"},
        ]
    )
    source.save_to_disk(str(tmp_path / "src"))
    server = HTTPServer(("127.0.0.1", 0), FakeTeacher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv(
            "TEACHER_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
        )
        monkeypatch.setenv("TEACHER_API_KEY", "test-key")
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "build-frontier-grpo-dataset.py",
                "--dataset",
                str(tmp_path / "src"),
                "--question_column",
                "q",
                "--answer_column",
                "a",
                "--teacher_model",
                "teacher",
                "--output_dir",
                str(tmp_path / "out"),
            ],
        )
        builder.main()
    finally:
        server.shutdown()

    out = datasets.load_from_disk(str(tmp_path / "out"))["train"]
    assert out.column_names == ["prompt", "solution", "steps"]
    assert len(out) == 1  # the wrong and the off-format teacher answers are dropped
    assert out[0]["solution"] == "\\boxed{42}"
    assert out[0]["steps"] == ["Six times seven is 42.", "So the product is 42."]
    assert "1/3 teacher solutions verified" in capsys.readouterr().out


def test_eval_task_prompt_matches_the_training_prompt_and_skips_the_reasoning_block():
    sys.path.insert(0, str(ROOT / "evals"))
    import unsolvable_math

    # The eval must ask exactly what the unguided training prompt asks.
    assert unsolvable_math.SYSTEM == builder.DEFAULT_SYSTEM
    assert unsolvable_math.SUFFIX == builder.DEFAULT_SUFFIX
    record = {"index": 7, "question": " What is 6*7? ", "ground_truth": "42"}
    sample = unsolvable_math.record_to_sample(record)
    row = builder.build_row(
        record["question"], "42", None, builder.DEFAULT_SYSTEM, builder.DEFAULT_SUFFIX
    )
    assert sample.input == row["prompt"][-1]["content"]
    assert sample.target == "42"
    # A box written while reasoning is not the answer.
    text = "<think>try \\boxed{1}</think>\nSo \\boxed{42}."
    assert unsolvable_math.answer_section(text) == "\nSo \\boxed{42}."
    assert (
        unsolvable_math.answer_section("no reasoning \\boxed{42}")
        == "no reasoning \\boxed{42}"
    )
