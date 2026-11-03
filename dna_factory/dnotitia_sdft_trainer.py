import logging
from functools import wraps

import torch
from trl.experimental.sdft import SDFTTrainer
from trl.experimental.sdft.sdft_trainer import DemonstrationTeacherContextBuilder

# Initialize logger
logger = logging.getLogger(__name__)

TEACHER_CONTEXT_PLACEMENTS = ("user", "system")


class PlacementTeacherContextBuilder(DemonstrationTeacherContextBuilder):
    """Builds the teacher prompt from the student prompt plus a per-row privileged context.

    `placement` decides where the context goes:
      - `user`:   TRL's default. The context is appended to the last user turn through
                  `teacher_prompt_template` (default `"{prompt}\\n\\n{privileged_context}"`).
      - `system`: The context becomes a system message in front of the conversation, so the teacher sees it
                  exactly as it would see a deployed system prompt.

    An empty (or missing) context leaves the teacher prompt identical to the student prompt under both
    placements. Such rows are anchor rows: the teacher is the plain frozen model, so the loss pulls the student
    toward its own initial behavior on that prompt.
    """

    def __init__(self, trainer, placement: str = "user"):
        super().__init__(trainer)
        if placement not in TEACHER_CONTEXT_PLACEMENTS:
            raise ValueError(
                f"`teacher_context_placement` must be one of {TEACHER_CONTEXT_PLACEMENTS}, got {placement!r}."
            )
        self.placement = placement

    def _compose_teacher_prompt(self, prompt, privileged_context):
        # TRL raises on `None`; treat it like an empty context so mixed datasets can leave the cell unset.
        text = (
            ""
            if privileged_context is None
            else self._stringify_privileged_context(privileged_context)
        )
        if not text.strip():
            return list(prompt) if isinstance(prompt, list) else prompt
        if self.placement == "system":
            if not isinstance(prompt, list):
                raise ValueError(
                    "`teacher_context_placement: system` needs conversational prompts (a list of messages)."
                )
            if prompt and prompt[0].get("role") == "system":
                # Chat templates accept one leading system message: merge instead of stacking a second one.
                merged = f"{text}\n\n{prompt[0].get('content', '')}"
                return [{"role": "system", "content": merged}, *prompt[1:]]
            return [{"role": "system", "content": text}, *prompt]
        return super()._compose_teacher_prompt(prompt, privileged_context)


class DnotitiaSDFTTrainer(SDFTTrainer):
    def __init__(
        self,
        *args,
        debug_first_n_batches: int = 3,
        teacher_context_placement: str = "user",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Maximum number of batches to print debug info for
        self.debug_first_n_batches = debug_first_n_batches
        # The stock builder is created in `SDFTTrainer.__init__` and only holds a back-reference to the trainer,
        # so it can be swapped afterwards.
        self.teacher_context_builder = PlacementTeacherContextBuilder(
            self, teacher_context_placement
        )

    def _compute_self_distillation_loss(self, model, inputs, distillation_logits):
        # TRL returns early, *before* its `accelerator.gather` of the logged loss, when a rank's micro-batch has no
        # loss tokens (every completion is empty/short, or all masked). Ranks that still have tokens do gather, so
        # the collectives fall out of step and NCCL hangs until the watchdog fires. Let the empty rank join the same
        # gather with a zero first, then take TRL's zero-loss path. A no-op in single-process runs.
        if distillation_logits.loss_mask.sum() == 0:
            self.accelerator.gather(
                torch.zeros((), device=distillation_logits.student_logits.device)
            )
        return super()._compute_self_distillation_loss(
            model, inputs, distillation_logits
        )

    def _decode_colored(self, ids, mask, color):
        text = ""
        for token_id, m in zip(ids, mask):
            # Use errors='replace' to handle incomplete UTF-8 sequences gracefully
            token_text = self.processing_class.decode(
                [int(token_id)], skip_special_tokens=False, errors="replace"
            ).replace("�", "🤗")
            text += f"\033[{color if m else 90}m{token_text}\033[0m"
        return text

    @wraps(SDFTTrainer.compute_loss)
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0

        if self._debug_count < self.debug_first_n_batches and "teacher_input_ids" in inputs:
            # First sample only: the student prompt, the teacher prompt (student prompt + privileged context in
            # the configured placement) and the shared on-policy completion.
            n_completion = inputs["completion_ids"].size(1)
            teacher_prompt_ids = inputs["teacher_input_ids"][0, :-n_completion]
            teacher_prompt_mask = inputs["teacher_attention_mask"][0, :-n_completion]
            logger.info("-" * 80)
            logger.info(
                f"TEACHER CONTEXT PLACEMENT: {self.teacher_context_builder.placement}"
            )
            logger.info(
                "INPUTS: \033[36mCYAN\033[0m for tokens included, \033[90mDARK GRAY\033[0m for padding, "
                "\033[33mYELLOW\033[0m for the completion tokens the loss is computed on, 🤗 for broken characters:"
            )
            logger.info(f"STUDENT PROMPT ({int(inputs['prompt_mask'][0].sum())} tokens):")
            logger.info(
                self._decode_colored(inputs["prompt_ids"][0], inputs["prompt_mask"][0], 36)
            )
            logger.info(f"TEACHER PROMPT ({int(teacher_prompt_mask.sum())} tokens):")
            logger.info(self._decode_colored(teacher_prompt_ids, teacher_prompt_mask, 36))
            logger.info("COMPLETION:")
            logger.info(
                self._decode_colored(
                    inputs["completion_ids"][0], inputs["completion_mask"][0], 33
                )
            )
            logger.info("-" * 80)
            self._debug_count += 1

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
