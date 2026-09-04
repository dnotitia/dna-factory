from dataclasses import dataclass, field


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
        metadata={"help": (
            "Handling of GRPO prompt groups whose rollouts all scored the same (zero advantage). "
            "'off': train on them as-is. "
            "'mask': skip their compute by truncating all-dead micro-batches to one token; "
            "leaves the gradient unchanged while beta=0 and no entropy/router-aux loss is used. "
            "'resample': refill the batch with informative groups from extra generation rounds, "
            "which changes the gradient by design. Not supported with streaming datasets."
        )}
    )
    dynamic_sampling_max_rounds: int = field(
        default=2,
        metadata={"help": "Extra generation rounds allowed by dynamic_sampling='resample'."}
    )
    periodic_save_seconds: float = field(
        default=0.0,
        metadata={"help": (
            "Wall-clock checkpoint interval in seconds (e.g. 21600 for 6 hours); "
            "0 disables it. The trainer only saves on step counts, so this callback "
            "sets should_save when the interval elapses and the normal save path "
            "(including save_total_limit rotation) handles the rest."
        )}
    )
