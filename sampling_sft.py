"""SFT on offline projection-sampled responses; see docs/sampling-sft.md."""

import sys
from dataclasses import replace

if __name__ == "__main__":
    from dna_factory.unsloth_bootstrap import maybe_import_unsloth

    maybe_import_unsloth(sys.argv[1:], "configs/_defaults-SamplingSFT.yaml")

from dna_factory.training_runner import cli_main
from sft import SPEC as SFT_SPEC

# The objective, preprocessing and ALL training defaults are ordinary SFT.
# Sampling happens offline with prepare_sampling_sft.py, before GPU training.
SPEC = replace(
    SFT_SPEC,
    name="SamplingSFT",
    output_dir_tag="SamplingSFT",
    script_file=__file__,
    defaults_yaml="configs/_defaults-SamplingSFT.yaml",
)

if __name__ == "__main__":
    cli_main(SPEC)
