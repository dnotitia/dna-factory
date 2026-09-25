from dataclasses import dataclass, field

from dna_factory.checkpoint_eval import DEFAULT_EVAL_TASKS


@dataclass
class DnotitiaArguments:
    """Arguments for Dnotitia SFT Trainer specific configurations"""

    debug_first_n_batches: int = field(
        default=3,
        metadata={
            "help": "Number of batches to print debug information for during training"
        },
    )
    dynamic_sampling: str = field(
        default="off",
        metadata={
            "help": (
                "Handling of GRPO prompt groups whose rollouts all scored the same (zero advantage). "
                "'off': train on them as-is. "
                "'mask': skip their compute by truncating all-dead micro-batches to one token; "
                "leaves the gradient unchanged while beta=0 and no entropy/router-aux loss is used. "
                "'resample': refill the batch with informative groups from extra generation rounds, "
                "which changes the gradient by design. Not supported with streaming datasets."
            )
        },
    )
    dynamic_sampling_max_rounds: int = field(
        default=2,
        metadata={
            "help": "Extra generation rounds allowed by dynamic_sampling='resample'."
        },
    )
    periodic_save_seconds: str = field(
        default="0",
        metadata={
            "help": (
                "Wall-clock checkpoint interval: human-friendly durations ('6h', '30m', "
                "'90s', '1d', combined '1h30m') or plain seconds ('21600'); "
                "'0'/'off' disables it. The trainer only saves on step counts, so this "
                "callback sets should_save when the interval elapses and the normal save "
                "path (including save_total_limit rotation) handles the rest. Works with "
                "save_strategy='no', which then leaves periodic saves as the only "
                "checkpoint source."
            )
        },
    )
    eval_on_checkpoint: bool = field(
        default=False,
        metadata={
            "help": (
                "Benchmark every checkpoint written during training: serve it with vLLM on "
                "eval_devices, run eval_tasks against it with Inspect, and log the scores to "
                "the live W&B run at the checkpoint's global_step. Runs in a background "
                "thread on rank 0 only -- training never waits for it, and an eval failure "
                "is a warning, never a crash. With the defaults below, turning this on is "
                "the only change needed; point eval_devices at GPUs training does not use."
            )
        },
    )
    eval_tasks: list[str] = field(
        default_factory=lambda: list(DEFAULT_EVAL_TASKS),
        metadata={
            "help": (
                "Inspect tasks to run on each checkpoint: registry names "
                "('inspect_evals/mmlu_pro') or local task files ('evals/kmmlu_pro.py', "
                "resolved against the repo root when cwd does not have them). Each task "
                "becomes one W&B metric, 'eval/<task>'."
            )
        },
    )
    eval_devices: str = field(
        default="0,1",
        metadata={
            "help": (
                "CUDA_VISIBLE_DEVICES for the eval vLLM server. Two GPUs by default, to "
                "match the --data-parallel-size 2 in eval_vllm_args: change one and change "
                "the other. Must not overlap the devices training runs on -- training "
                "already owns its GPUs, and a server sharing them OOMs the run."
            )
        },
    )
    eval_vllm_args: str = field(
        default=("--max-model-len 32768 --data-parallel-size 2"),
        metadata={
            "help": (
                "Extra flags appended to `vllm serve <checkpoint>` for the eval server. "
                "Data parallel by default: one full replica per GPU, which is what an eval "
                "wants (many independent samples) -- use --tensor-parallel-size instead for "
                "a model too big for one GPU. The parallel sizes must multiply out to the "
                "number of devices in eval_devices. A --port here pins the port; otherwise "
                "a free one is picked from 8000 up."
            )
        },
    )
    eval_max_connections: int = field(
        default=20,
        metadata={
            "help": (
                "`inspect eval --max-connections`: concurrent requests in flight against "
                "the eval server."
            )
        },
    )
    eval_max_tokens: int = field(
        default=16000,
        metadata={"help": "`inspect eval --max-tokens`: generation cap per sample."},
    )
