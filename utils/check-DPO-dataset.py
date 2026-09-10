import sys
import json
from datasets import load_dataset

# Color codes
YELLOW = '\033[93m'
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

'''
# Type-1 Conversion

## AS-IS
>>> dataset[0]
{
  'chosen': 'Certainly! Below is a sample of a waterfall clause...',
  'rejected': '## Waterfall Clause Example for an Operating Agreement...',
  'prompt': 'Include keywords sireless, tour in your response can you write me a waterfall clause for an operating agreement'
}

## TO-BE
>>> dataset[0]
{
  'chosen': [
    {'content': 'Include keywords sireless, tour in your response can you write me a waterfall clause for an operating agreement', 'role': 'user'},
    {'content': 'Certainly! Below is a sample of a waterfall clause...', 'role': 'assistant'}
  ],
  'rejected': [
    {'content': 'Include keywords sireless, tour in your response can you write me a waterfall clause for an operating agreement', 'role': 'user'},
    {'content': '## Waterfall Clause Example for an Operating Agreement...', 'role': 'assistant'}
  ]
}
'''


def validate_messages_field(messages, row_idx, field_name):
    """
    Validates a list of messages follows the correct format:
    - User messages: {'content': str, 'role': 'user'}
    - Assistant messages: {'content': str, 'role': 'assistant'}
    
    Returns: (is_valid, error_message)
    """
    if not isinstance(messages, list) or len(messages) == 0:
        print(f"{YELLOW}Messages data:{RESET} {json.dumps(messages, indent=2, ensure_ascii=False)}")
        return False, f"Row {row_idx}, Field '{field_name}': Not a valid list or empty"

    for msg_idx, message in enumerate(messages):
        if not isinstance(message, dict):
            print(f"{YELLOW}Message data:{RESET} {message}")
            return False, f"Row {row_idx}, Field '{field_name}', Message {msg_idx}: Not a dictionary"

        # Check required fields
        if 'content' not in message or 'role' not in message:
            print(f"{YELLOW}Message data:{RESET} {json.dumps(message, indent=2, ensure_ascii=False)}")
            return False, f"Row {row_idx}, Field '{field_name}', Message {msg_idx}: Missing 'content' or 'role'"

        # Check that content is not empty
        if not message['content'] or not isinstance(message['content'], str):
            print(
                f"\n[WARNING] Row {row_idx}, Field '{field_name}', Message {msg_idx}: 'content' is empty or not a string")
            print(f"{YELLOW}Message data:{RESET} {json.dumps(message, indent=2, ensure_ascii=False)}")

        role = message['role']

        # Validate role
        if role not in ['user', 'assistant']:
            print(f"{YELLOW}Message data:{RESET} {json.dumps(message, indent=2, ensure_ascii=False)}")
            return False, f"Row {row_idx}, Field '{field_name}', Message {msg_idx}: Invalid role '{role}' (must be 'user' or 'assistant')"

    return True, None


def validate_dataset(dataset):
    """
    Validates that the dataset has 'chosen' and 'rejected' columns and both follow the correct format:
    - Each should be a list of messages
    - User messages: {'content': str, 'role': 'user'}
    - Assistant messages: {'content': str, 'role': 'assistant'}
    """
    # Check if 'chosen' and 'rejected' columns exist
    if 'chosen' not in dataset.column_names:
        return False, "Missing 'chosen' column", None

    if 'rejected' not in dataset.column_names:
        return False, "Missing 'rejected' column", None

    # Check each example for correct format
    for idx, example in enumerate(dataset):
        chosen = example.get('chosen')
        rejected = example.get('rejected')

        # Validate 'chosen' field
        is_valid, error_msg = validate_messages_field(chosen, idx, 'chosen')
        if not is_valid:
            print(f"{YELLOW}Complete data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
            return False, error_msg, example

        # Validate 'rejected' field
        is_valid, error_msg = validate_messages_field(rejected, idx, 'rejected')
        if not is_valid:
            print(f"{YELLOW}Complete data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
            return False, error_msg, example

        # Optional: Check that chosen and rejected have the same user prompts
        if len(chosen) > 0 and len(rejected) > 0:
            if chosen[0].get('role') == 'user' and rejected[0].get('role') == 'user':
                if chosen[0].get('content') != rejected[0].get('content'):
                    print(f"\n[WARNING] Row {idx}: 'chosen' and 'rejected' have different user prompts")
                    print(f"{YELLOW}Chosen prompt:{RESET} {chosen[0].get('content')[:100]}...")
                    print(f"{YELLOW}Rejected prompt:{RESET} {rejected[0].get('content')[:100]}...")

    return True, "All checks passed", None


def is_valid_messages(messages):
    """Check if all messages have non-empty content"""
    if not isinstance(messages, list):
        return False
    for message in messages:
        content = message.get('content', '')
        if not content or content.strip() == '':
            return False
    return True


def transform_type1(example):
    """Transform plain text chosen/rejected to message format with prompt"""
    prompt = example.get('prompt', '')
    chosen_text = example.get('chosen', '')
    rejected_text = example.get('rejected', '')

    # Create chosen messages
    chosen_messages = [
        {'content': prompt, 'role': 'user'},
        {'content': chosen_text, 'role': 'assistant'}
    ]

    # Create rejected messages
    rejected_messages = [
        {'content': prompt, 'role': 'user'},
        {'content': rejected_text, 'role': 'assistant'}
    ]

    return {
        'chosen': chosen_messages,
        'rejected': rejected_messages
    }


def filter_valid_data(example):
    """Filter out examples with empty content"""
    prompt = example.get('prompt', '')
    chosen = example.get('chosen', '')
    rejected = example.get('rejected', '')
    return (isinstance(prompt, str) and prompt.strip() != '' and
            isinstance(chosen, str) and chosen.strip() != '' and
            isinstance(rejected, str) and rejected.strip() != '')


def check_mode(dataset_name):
    """Check mode: validate dataset format"""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split='train')

    print("Validating dataset format...")
    is_valid, message, error_example = validate_dataset(dataset)

    if is_valid:
        print(f"\n{GREEN}✓ Valid{RESET}")
        print(f"Total examples validated: {len(dataset)}")
    else:
        print(f"\n{RED}✗ Not Valid{RESET}")
        print(f"Total examples: {len(dataset)}")
        print(f"Reason: {message}")
        sys.exit(1)


def convert_mode(source_dataset, target_dataset, conversion_type):
    """Convert mode: transform and push dataset"""
    print(f"Loading dataset: {source_dataset}")
    dataset = load_dataset(source_dataset, split='train')
    print(f"Original dataset size: {len(dataset)}")

    # Filter out invalid data first
    print("Filtering valid data...")
    dataset = dataset.filter(filter_valid_data, desc="Filtering valid data")
    print(f"After filtering: {len(dataset)}")

    # Apply transformation based on type
    if conversion_type == 'type1':
        print("Applying type1 conversion...")
        dataset = dataset.map(transform_type1, remove_columns=['prompt'],
                              desc="Converting to DPO message format (type1)")
    else:
        print(f"Error: Unknown conversion type '{conversion_type}'")
        print("Available types: type1")
        sys.exit(1)

    # Push to hub
    print(f"Pushing to hub: {target_dataset}")
    dataset.push_to_hub(target_dataset, private=True)
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Check mode: python check-DPO-dataset.py <dataset_name>
        dataset_name = sys.argv[1]
        check_mode(dataset_name)
    elif len(sys.argv) == 4:
        # Convert mode: python check-DPO-dataset.py <source_dataset> <target_dataset> <conversion_type>
        source_dataset = sys.argv[1]
        target_dataset = sys.argv[2]
        conversion_type = sys.argv[3]
        convert_mode(source_dataset, target_dataset, conversion_type)
    else:
        print("Usage:")
        print("  Check mode:   python check-DPO-dataset.py <dataset_name>")
        print("  Convert mode: python check-DPO-dataset.py <source_dataset> <target_dataset> <conversion_type>")
        sys.exit(1)
