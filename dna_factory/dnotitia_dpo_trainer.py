import logging
from functools import wraps

from torch import nn
from trl import DPOTrainer

# Initialize logger
logger = logging.getLogger(__name__)


class DnotitiaDPOTrainer(DPOTrainer):
    def __init__(self, *args, debug_first_n_batches: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        # Maximum number of batches to print debug info for
        self.debug_first_n_batches = debug_first_n_batches

    @wraps(DPOTrainer.compute_loss)
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # TRL's preference collator combines chosen and rejected sequences into
        # one batch: [chosen_0, ..., chosen_n, rejected_0, ..., rejected_n].
        # `completion_mask` distinguishes prompt tokens (0) from completion
        # tokens (1). Older TRL releases exposed separate prompt/chosen/rejected
        # input-id fields, so do not rely on those keys here.
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0

        required_keys = {"input_ids", "attention_mask", "completion_mask"}
        if self._debug_count < self.debug_first_n_batches and required_keys.issubset(
            inputs
        ):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            completion_mask = inputs["completion_mask"]
            pair_count = len(input_ids) // 2

            def decode_colored(ids, attention, completion, rejected=False):
                text = ""
                for token_id, is_attended, is_completion in zip(
                    ids, attention, completion
                ):
                    token_text = self.processing_class.decode(
                        [token_id.item()], skip_special_tokens=False, errors="replace"
                    ).replace("�", "🤗")
                    if is_attended.item() == 0:
                        color = "90"  # Dark gray: padding
                    elif rejected and is_completion.item() == 1:
                        color = "91"  # Red: rejected completion
                    else:
                        color = "36"  # Cyan: prompt or chosen completion
                    text += f"\033[{color}m{token_text}\033[0m"
                return text

            for index in range(pair_count):
                chosen_ids = input_ids[index]
                chosen_attention = attention_mask[index]
                chosen_completion = completion_mask[index]
                rejected_ids = input_ids[index + pair_count]
                rejected_attention = attention_mask[index + pair_count]
                rejected_completion = completion_mask[index + pair_count]

                prompt_length = int(
                    ((chosen_attention == 1) & (chosen_completion == 0)).sum().item()
                )
                chosen_length = int(
                    ((chosen_attention == 1) & (chosen_completion == 1)).sum().item()
                )
                rejected_length = int(
                    ((rejected_attention == 1) & (rejected_completion == 1))
                    .sum()
                    .item()
                )

                logger.info("-" * 80)
                logger.info(f"PROMPT LENGTH: {prompt_length:,}")
                logger.info(f"CHOSEN LENGTH: {chosen_length:,}")
                logger.info(f"REJECTED LENGTH: {rejected_length:,}")
                logger.info(
                    "INPUTS: \033[36mCYAN\033[0m for prompt/chosen tokens, \033[91mRED\033[0m for rejected tokens, "
                    "\033[90mDARK GRAY\033[0m when attention_mask is 0 (means padding), 🤗 for broken characters "
                    "from multi-byte decoding:"
                )
                logger.info("-" * 80)
                logger.info(
                    f"CHOSEN: {decode_colored(chosen_ids, chosen_attention, chosen_completion)}"
                )
                logger.info(
                    f"REJECTED: {decode_colored(rejected_ids, rejected_attention, rejected_completion, rejected=True)}"
                )
                logger.info("-" * 80)

            self._debug_count += 1

        # TRL's Liger DPO implementation accesses ``model.base_model``
        # directly. Data-parallel wrappers hide that attribute, so use TRL's
        # regular DPO implementation for wrapped models. It executes through
        # the wrapper's forward pass and therefore preserves parallel training.
        # Liger remains enabled for unwrapped models.
        use_liger_with_data_parallel = self.use_liger_kernel and isinstance(
            model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
        )
        if use_liger_with_data_parallel:
            if not getattr(self, "_warned_liger_data_parallel_fallback", False):
                logger.warning(
                    "Liger DPO is incompatible with a data-parallel model wrapper in "
                    "the installed TRL version; using standard DPO loss for this run. "
                    "Liger remains enabled for unwrapped models."
                )
                self._warned_liger_data_parallel_fallback = True

            original_use_liger_kernel = self.use_liger_kernel
            self.use_liger_kernel = False
            try:
                loss_output = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            finally:
                self.use_liger_kernel = original_use_liger_kernel
        else:
            # Call parent class's compute_loss method to calculate the actual loss.
            loss_output = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        if return_outputs:
            # If return_outputs is True, return the full output (loss, outputs)
            return loss_output
        else:
            # If return_outputs is False, extract and return the first element (loss)
            if isinstance(loss_output, tuple):
                return loss_output[0]
            return loss_output
