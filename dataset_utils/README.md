# DNA Factory - Dataset Utils

Utilities for validating and converting datasets used in DNA Factory for both SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) training.

---

**SFT Dataset Tool** (`check-SFT-dataset.py`)

A comprehensive tool for validating and converting SFT datasets with thinking format support. This tool ensures your datasets meet DNA Factory's requirements and can automatically transform legacy formats into the modern thinking-augmented format.

**Dataset Format**

DNA Factory's SFT training uses the following message structure:

- **User messages**: `{'content': str, 'role': 'user'}`
  - Contains the user's question or prompt
  
- **Assistant messages**: `{'content': str, 'role': 'assistant', 'thinking': str}`
  - `content`: The final response visible to users
  - `thinking`: The model's internal reasoning process (optional)

The `thinking` field enables the model to learn step-by-step reasoning patterns, improving its problem-solving capabilities.

**Check Mode (Validation)**

Validates that your dataset follows the correct format. The tool performs comprehensive checks including:
- Verifying the structure of each example
- Ensuring all required fields are present
- Validating data types for each field

Usage:
```bash
$ python check-SFT-dataset.py <dataset_name>
```

Example:
```bash
$ python check-SFT-dataset.py dnotitia/Reasoning_R1_Kor_completion_25k_sharegpt_v2
Loading dataset: dnotitia/Reasoning_R1_Kor_completion_25k_sharegpt_v2
Validating dataset format...

✓ Valid
Total examples validated: 24575
```

**Convert Mode (Transformation)**

Automatically converts legacy dataset formats into DNA Factory's thinking format. The converted dataset is uploaded directly to Hugging Face Hub, making it immediately available for training.

Usage:
```bash
$ python check-SFT-dataset.py <source_dataset> <target_dataset> <conversion_type>
```

Example:
```bash
$ python check-SFT-dataset.py dnotitia/old-dataset dnotitia/new-dataset type1
```

**Conversion Types:**

- **`type1`**: 
  - Source field: `messages` (already in role/content format)
  - Extracts thinking content wrapped in `<think>...</think>` tags and separates it into a dedicated field
  - Removes `<answer>...</answer>` tags from assistant responses (keeps only the content inside)
  - Removes `/think` or `/no_think` suffixes from user prompts
  - Example transformation: `"What is 2+2? /think"` → `"What is 2+2?"`
  
- **`type2`**:
  - Source field: `conversations` (from/value format, ShareGPT style)
  - Converts `conversations` format to `messages` format (`from` → `role`, `value` → `content`)
  - Performs the same thinking extraction, answer tag removal, and suffix removal as type1
  - Ideal for **ShareGPT-formatted datasets** that need full restructuring

---

**DPO Dataset Tool** (`check-DPO-dataset.py`)

A specialized tool for validating and converting DPO datasets with chosen/rejected message pairs. DPO (Direct Preference Optimization) trains models to align with human preferences by learning from response comparisons.

**Dataset Format**

DPO training uses preference pairs (chosen/rejected) to align model behavior:

```python
{
  'chosen': [
    {'content': str, 'role': 'user'},
    {'content': str, 'role': 'assistant'}
  ],
  'rejected': [
    {'content': str, 'role': 'user'},
    {'content': str, 'role': 'assistant'}
  ]
}
```

- **chosen**: The conversation containing the preferred (good) response
- **rejected**: The conversation containing the non-preferred (bad) response
- Both conversations start with the same user message; only the assistant responses differ

This format teaches the model which types of responses are more desirable, improving response quality and alignment with user expectations.

**Check Mode (Validation)**

Validates that your DPO dataset has the correct format. The tool verifies:
- Presence of both chosen and rejected pairs
- Correct message structure in both conversations

Usage:
```bash
$ python check-DPO-dataset.py <dataset_name>
```

Example:
```bash
$ python check-DPO-dataset.py dnotitia/dpo_claude3.5-sonnet_15k_v4
Loading dataset: dnotitia/dpo_claude3.5-sonnet_15k_v4
Validating dataset format...

✓ Valid
Total examples validated: 15023
```

**Convert Mode (Transformation)**

Converts legacy prompt-based formats to the modern message-based format. The converted dataset is automatically uploaded to Hugging Face Hub for immediate use.

Usage:
```bash
$ python check-DPO-dataset.py <source_dataset> <target_dataset> <conversion_type>
```

Example:
```bash
$ python check-DPO-dataset.py dnotitia/old-dpo-dataset dnotitia/new-dpo-dataset type1
```

**Conversion Types:**

- **`type1`**: 
  - Source format: `{prompt: str, chosen: str, rejected: str}`
  - Target format: Message-based format (see above)
  - Converts `prompt` to user message, `chosen`/`rejected` to respective assistant messages

---

Required packages:
```bash
$ pip install datasets
```

Hugging Face Authentication:

To access private datasets or upload converted datasets, authenticate with Hugging Face:

```bash
$ huggingface-cli login
```

After login, provide a token with write permissions. This enables automatic dataset uploads during conversion operations.
