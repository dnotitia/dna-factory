"""
Test cases for print_environment_and_arguments in dna_factory/training_runner.py

Pins the exact sequence of strings passed to `_logger.info`.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.training_runner import print_environment_and_arguments

SEPARATOR = "-" * 90
YELLOW = "\033[33m"
PINK = "\033[1;95m"
RESET = "\033[0m"


class _FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)


def test_logged_lines(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "1")
    logger = _FakeLogger()

    print_environment_and_arguments(
        logger,
        SimpleNamespace(dataset_name="ds"),
        SimpleNamespace(learning_rate=0.1),
        SimpleNamespace(model_name_or_path="m"),
        SimpleNamespace(datasets=[]),
        SimpleNamespace(use_liger=False),
        {"learning_rate"},
    )

    assert logger.lines == [
        SEPARATOR,
        " OS ENVIRONMENT VARIABLES:",
        SEPARATOR,
        f"{YELLOW}CUDA_VISIBLE_DEVICES: 0,1{RESET}",
        "WORLD_SIZE: None",
        "RANK: 0",
        f"{YELLOW}LOCAL_RANK: 1{RESET}",
        SEPARATOR,
        " SCRIPT ARGUMENTS:",
        SEPARATOR,
        "{'dataset_name': 'ds'}",
        SEPARATOR,
        " TRAINING ARGUMENTS:",
        SEPARATOR,
        f"{PINK}{{'learning_rate': 0.1}}{RESET}",
        SEPARATOR,
        " MODEL ARGUMENTS:",
        SEPARATOR,
        "{'model_name_or_path': 'm'}",
        SEPARATOR,
        " DATASET MIXTURE ARGUMENTS:",
        SEPARATOR,
        "{'datasets': []}",
        SEPARATOR,
        " DNOTITIA TRAINER ARGUMENTS:",
        SEPARATOR,
        "{'use_liger': False}",
        SEPARATOR,
        "",
    ]
