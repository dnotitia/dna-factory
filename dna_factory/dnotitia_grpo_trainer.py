import logging
from contextlib import contextmanager
from functools import wraps

from trl import GRPOTrainer

from dna_factory.dynamic_sampling import DynamicSamplingMixin

# Initialize logger
logger = logging.getLogger(__name__)


class DnotitiaGRPOTrainer(DynamicSamplingMixin, GRPOTrainer):
    """GRPOTrainer with colored token debug output.

    Dynamic sampling comes from DynamicSamplingMixin; see dna_factory/dynamic_sampling.py.
    """

    def __init__(
        self,
        *args,
        debug_first_n_batches: int = 3,
        log_completions_steps: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Maximum number of batches to print debug info for
        self.debug_first_n_batches = debug_first_n_batches
        # 0 keeps TRL's cadence: the completion table rides along with every log() call.
        self.log_completions_steps = max(int(log_completions_steps or 0), 0)
        self._last_completion_log_step = 0
        if self.log_completions and self.log_completions_steps:
            logger.info(
                "Completion table (W&B + %s/completions) every %d steps; "
                "scalar metrics still follow logging_steps=%s.",
                self.args.output_dir,
                self.log_completions_steps,
                self.args.logging_steps,
            )

    @contextmanager
    def _completion_log_scope(self):
        """Hide log_completions from TRL's log() until the table interval is due.

        TRL writes the W&B table and the local parquet in the same block, gated only on
        this flag, and log() itself runs every logging_steps. Scalar metrics go out
        through Trainer.log before that block, so suppressing the flag leaves them alone.
        """
        interval = self.log_completions_steps
        if not self.log_completions or interval <= 0:
            yield
            return
        step = int(self.state.global_step)
        if step - self._last_completion_log_step < interval:
            self.log_completions = False
            try:
                yield
            finally:
                self.log_completions = True
            return
        yield
        self._last_completion_log_step = step

    @wraps(GRPOTrainer.log)
    def log(self, logs, start_time=None):
        with self._completion_log_scope():
            super().log(logs, start_time)

    @wraps(GRPOTrainer.compute_loss)
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Decode and display prompt/completion ids with mask highlighting
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0

        for i in range(len(inputs["prompt_ids"])):
            if (
                self._debug_count < self.debug_first_n_batches
                and "prompt_ids" in inputs
                and "prompt_mask" in inputs
                and "completion_ids" in inputs
                and "completion_mask" in inputs
                and "advantages" in inputs
            ):
                # Take i-th sample in batch
                prompt_ids = inputs["prompt_ids"][i]
                prompt_masks = inputs["prompt_mask"][i]
                completion_ids = inputs["completion_ids"][i]
                completion_masks = inputs["completion_mask"][i]
                advantage = inputs["advantages"][i]

                prompt_colored_text = ""
                for prompt_id, prompt_mask in zip(prompt_ids, prompt_masks):
                    # Use errors='replace' to handle incomplete UTF-8 sequences gracefully
                    # For multi-byte characters like Korean, replace the � character with 🤗
                    token_text = self.processing_class.decode(
                        [prompt_id], skip_special_tokens=False, errors="replace"
                    )
                    token_text = token_text.replace("�", "🤗")

                    if prompt_mask == 0:
                        prompt_colored_text += f"\033[90m{token_text}\033[0m"  # Dark gray when prompt_mask is 0
                    else:
                        prompt_colored_text += (
                            f"\033[36m{token_text}\033[0m"  # Cyan color
                        )

                completion_colored_text = ""
                for completion_id, completion_mask in zip(
                    completion_ids, completion_masks
                ):
                    # Use errors='replace' to handle incomplete UTF-8 sequences gracefully
                    # For multi-byte characters like Korean, replace the � character with 🤗
                    token_text = self.processing_class.decode(
                        [completion_id], skip_special_tokens=False, errors="replace"
                    )
                    token_text = token_text.replace("�", "🤗")

                    if completion_mask == 0:
                        completion_colored_text += f"\033[90m{token_text}\033[0m"  # Dark gray when completion_mask is 0
                    else:
                        completion_colored_text += (
                            f"\033[36m{token_text}\033[0m"  # Cyan color
                        )

                # Advantage is a per-sample scalar at this point (group-normalized reward)
                advantage_value = (
                    advantage.item() if hasattr(advantage, "item") else float(advantage)
                )

                logger.info("-" * 80)
                logger.info(f"PROMPT LENGTH: {len(prompt_ids):,}")
                logger.info(f"COMPLETION LENGTH: {len(completion_ids):,}")
                logger.info(f"ADVANTAGE: {advantage_value:+.6f}")
                logger.info(
                    "INPUTS: \033[36mCYAN\033[0m for prompt/completion tokens included in loss, "
                    "\033[90mDARK GRAY\033[0m when mask is 0 (means padding/masked-out), 🤗 for broken characters "
                    "from multi-byte decoding:"
                )
                logger.info("-" * 80)
                logger.info(f"PROMPT: {prompt_colored_text}")
                logger.info(f"COMPLETION: {completion_colored_text}")
                logger.info("-" * 80)

                self._debug_count += 1

        # Call parent class's compute_loss method to calculate the actual loss.
        # Note: GRPOTrainer.compute_loss raises if return_outputs=True, so it is never forwarded.
        return super().compute_loss(
            model, inputs, num_items_in_batch=num_items_in_batch
        )
