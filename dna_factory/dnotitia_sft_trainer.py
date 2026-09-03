import logging
from functools import wraps

from trl import SFTTrainer

# Initialize logger
logger = logging.getLogger(__name__)


class DnotitiaSFTTrainer(SFTTrainer):
    def __init__(self, *args, debug_first_n_batches: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        # Maximum number of batches to print debug info for
        self.debug_first_n_batches = debug_first_n_batches

    @wraps(SFTTrainer.compute_loss)
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Decode and display input_ids with labels highlighting
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0

        for i in range(len(inputs["input_ids"])):
            if (
                self._debug_count < self.debug_first_n_batches
                and "input_ids" in inputs
                and "labels" in inputs
            ):
                # Take i-th sample in batch
                input_ids = inputs["input_ids"][i]
                labels = inputs["labels"][i]
                position_ids = inputs.get("position_ids", None)
                if position_ids is not None:
                    position_ids = position_ids[i]

                # Create colored text by comparing input_ids and labels
                colored_text = ""
                for i, (input_id, label_id) in enumerate(zip(input_ids, labels)):
                    # Use errors='replace' to handle incomplete UTF-8 sequences gracefully
                    # For multi-byte characters like Korean, replace the � character with 🤗
                    token_text = self.processing_class.decode(
                        [input_id], skip_special_tokens=False, errors="replace"
                    )
                    token_text = token_text.replace("�", "🤗")

                    # Priority: if position_ids resets to 0 at this position, color YELLOW
                    is_pos_reset = False
                    if position_ids is not None:
                        pos_val = position_ids[i]
                        if hasattr(pos_val, "item"):
                            pos_val = pos_val.item()
                        is_pos_reset = pos_val == 0

                    if is_pos_reset:
                        colored_text += f"\033[93m{token_text}\033[0m"  # Yellow when position_ids resets to 0
                    elif label_id == input_id:
                        colored_text += f"\033[36m{token_text}\033[0m"  # Cyan color
                    else:
                        colored_text += f"\033[91m{token_text}\033[0m"  # Red color

                logger.info("-" * 80)
                logger.info(f"LENGTH: {len(input_ids):,}")
                logger.info(
                    "INPUTS: \033[93mYELLOW\033[0m when position_ids reset to 0, \033[36mCYAN\033[0m for tokens "
                    "included in loss, \033[91mRED\033[0m for tokens excluded from loss, 🤗 for broken characters "
                    "from multi-byte decoding:"
                )
                logger.info("-" * 80)
                logger.info(colored_text)
                logger.info("-" * 80)

                self._debug_count += 1

        # Call parent class's compute_loss method to calculate the actual loss
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
