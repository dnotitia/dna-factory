# Utils

Dataset validation/conversion for SFT and DPO, plus a vLLM compatibility check for GRPO.

## check-SFT-dataset.py

Target format: user `{'content', 'role'}` + assistant `{'content', 'role', 'thinking'}` (`thinking` optional).

```bash
# Validate
python check-SFT-dataset.py <dataset_name>

# Convert legacy format and upload to Hub (requires `huggingface-cli login`)
python check-SFT-dataset.py <source_dataset> <target_dataset> <type>
```

Conversion types:

- `type1`: source `messages` (role/content). Extracts `<think>...</think>` into `thinking`, strips `<answer>` tags, removes `/think`/`/no_think` suffixes from prompts.
- `type2`: source `conversations` (ShareGPT from/value). Same cleanup as type1, plus `from`→`role`, `value`→`content`.

## check-DPO-dataset.py

Target format: `{chosen: [...], rejected: [...]}` — both start with the same user message, assistant responses differ.

```bash
# Validate
python check-DPO-dataset.py <dataset_name>

# Convert legacy format and upload to Hub (requires `huggingface-cli login`)
python check-DPO-dataset.py <source_dataset> <target_dataset> <type>
```

Conversion types:

- `type1`: source `{prompt, chosen, rejected}` (plain strings) → message-based format.

## check-vllm-version.sh

Shows the vLLM range your installed TRL supports (needed for GRPO `--use_vllm true`).

```bash
source .venv/bin/activate  # venv python required
bash utils/check-vllm-version.sh
```

If installed vLLM is outside the range (`uv pip show vllm`), pin vLLM into it or fall back to `--use_vllm false`.
