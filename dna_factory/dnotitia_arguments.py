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
