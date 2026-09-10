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
>>> dataset[0]['messages']
[
  {
    'role': 'user',
    'content': '당신은 누구입니까? /think'
  }, 
  {
    'role': 'assistant',
    'content': '<think>\nOkay, I need to answer the question "당신은 누구입니까?" which means "Who are you?" in Korean.\n</think>\n\n저는 **DNA 2.0**입니다. 디노티시아(Dnotitia Inc.)에서 개발한 최신 대형 언어 모델(LLM)로, 한국어와 영어 처리에 최적화된 이중 언어 특화 모델입니다.'
  }
]

## TO-BE
>>> dataset[0]['messages']
[
  {
    'role': 'user',
    'content': '당신은 누구입니까?'
  }, 
  {
    'role': 'assistant',
    'thinking': 'Okay, I need to answer the question "당신은 누구입니까?" which means "Who are you?" in Korean.',
    'content': '저는 **DNA 2.0**입니다. 디노티시아(Dnotitia Inc.)에서 개발한 최신 대형 언어 모델(LLM)로, 한국어와 영어 처리에 최적화된 이중 언어 특화 모델입니다.'
  }
]

# Type-2 Conversion
## AS-IS
>>> dataset[0]['conversations']
[
  {
    'from': 'user',
    'value': '당신은 누구입니까? /think'
  }, 
  {
    'from': 'assistant',
    'value': '<think>\nOkay, I need to answer the question "당신은 누구입니까?" which means "Who are you?" in Korean.\n</think>\n\n저는 **DNA 2.0**입니다. 디노티시아(Dnotitia Inc.)에서 개발한 최신 대형 언어 모델(LLM)로, 한국어와 영어 처리에 최적화된 이중 언어 특화 모델입니다.'
  }
]

## TO-BE
>>> dataset[0]['messages']
[
  {
    'role': 'user',
    'content': '당신은 누구입니까?'
  }, 
  {
    'role': 'assistant',
    'thinking': 'Okay, I need to answer the question "당신은 누구입니까?" which means "Who are you?" in Korean.',
    'content': '저는 **DNA 2.0**입니다. 디노티시아(Dnotitia Inc.)에서 개발한 최신 대형 언어 모델(LLM)로, 한국어와 영어 처리에 최적화된 이중 언어 특화 모델입니다.'
  }
]

'''


def validate_dataset(dataset):
    """
    Validates that the dataset has 'messages' column and follows the correct format:
    - System messages: {'content': str, 'role': 'system'}
    - User messages: {'content': str, 'role': 'user'}
    - Assistant messages: {'content': str, 'role': 'assistant', 'thinking': str}
    """
    # Check if 'messages' column exists
    if 'messages' not in dataset.column_names:
        return False, "Missing 'messages' column", None

    # Check a sample of messages for correct format
    for idx, example in enumerate(dataset):
        messages = example.get('messages')

        if not isinstance(messages, list) or len(messages) == 0:
            print(f"{YELLOW}Example data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
            return False, f"Row {idx}: 'messages' is not a valid list", example

        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                print(f"{YELLOW}Message data:{RESET} {message}")
                print(f"{YELLOW}Complate data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
                return False, f"Row {idx}, Message {msg_idx}: Not a dictionary", example

            # Check required fields
            if 'content' not in message or 'role' not in message:
                print(f"{YELLOW}Message data:{RESET} {json.dumps(message, indent=2, ensure_ascii=False)}")
                print(f"{YELLOW}Complate data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
                return False, f"Row {idx}, Message {msg_idx}: Missing 'content' or 'role'", example

            # Check that content is not empty
            if not message['content'] or not isinstance(message['content'], str):
                print(f"\n[WARNING] Row {idx}, Message {msg_idx}: 'content' is empty or not a string")
                print(f"{YELLOW}Message data:{RESET} {json.dumps(message, indent=2, ensure_ascii=False)}")
                print(f"{YELLOW}Complate data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")

            role = message['role']

            # Validate based on role
            if role == 'user' or role == 'system':
                # User and system messages should not have 'thinking' unless it's null
                if 'thinking' in message and message['thinking'] is not None:
                    print(f"{YELLOW}Message data:{RESET} {json.dumps(message, indent=2, ensure_ascii=False)}")
                    print(f"{YELLOW}Complate data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
                    return False, f"Row {idx}, Message {msg_idx}: {role.capitalize()} message should not have 'thinking' field (unless null)", example
            elif role == 'assistant':
                # Assistant messages should have 'thinking' field (value can be empty or null)
                if 'thinking' not in message:
                    print(f"{YELLOW}Message data:{RESET} {json.dumps(message, indent=2, ensure_ascii=False)}")
                    print(f"{YELLOW}Complate data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
                    return False, f"Row {idx}, Message {msg_idx}: Assistant message missing 'thinking' field", example
            else:
                print(f"{YELLOW}Message data:{RESET} {json.dumps(message, indent=2, ensure_ascii=False)}")
                print(f"{YELLOW}Complate data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
                return False, f"Row {idx}, Message {msg_idx}: Invalid role '{role}' (must be 'system', 'user' or 'assistant')", example

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

def filter_valid_data(example):
    """Filter out examples with empty content"""
    return is_valid_messages(example.get('messages'))


def transform_type1(example):
    """Transform messages format for type1 conversion"""
    if 'messages' not in example:
        raise ValueError(f"'messages' field not found in example.")
    
    messages = example['messages']
    if isinstance(messages, list):
        new_messages = []
        for message in messages:
            role = message.get('role')
            content = message.get('content', '')
            updated = dict(message)
            if role == 'user':
                updated['content'] = content.split(' /think')[0].split(' /no_think')[0]
                # Ensure 'thinking' field does not exist on user messages
                if 'thinking' in updated:
                    updated.pop('thinking', None)
            elif role == 'assistant':
                if '<think>' in content and '</think>' in content:
                    thinking_part = content.split('<think>')[1].split('</think>')[0].strip()
                    content_part = content.split('</think>')[1].strip()
                    # Remove <answer> tags if present
                    if '<answer>' in content_part and '</answer>' in content_part:
                        content_part = content_part.split('<answer>')[1].split('</answer>')[0].strip()
                    updated['thinking'] = thinking_part
                    updated['content'] = content_part
            new_messages.append(updated)
        return {'messages': new_messages}
    return {'messages': messages}


def transform_type2(example):
    """Transform conversations format to messages format for type2 conversion"""
    if 'conversations' not in example:
        raise ValueError(f"'conversations' field not found in example.")
    
    conversations = example['conversations']
    if isinstance(conversations, list):
        new_messages = []
        for conv in conversations:
            # Convert 'from' to 'role' and 'value' to 'content'
            from_field = conv.get('from')
            value = conv.get('value', '')
            
            # Map 'from' field to 'role' field
            # Common mappings: 'human'/'user' -> 'user', 'gpt'/'assistant' -> 'assistant'
            if from_field in ['human', 'user']:
                role = 'user'
            elif from_field in ['gpt', 'assistant']:
                role = 'assistant'
            elif from_field == 'system':
                role = 'system'
            else:
                role = from_field  # Keep as is if unknown
            
            updated = {'role': role, 'content': value}
            
            if role == 'user':
                # Remove /think and /no_think suffixes
                updated['content'] = value.split(' /think')[0].split(' /no_think')[0]
            elif role == 'assistant':
                # Process thinking tags
                if '<think>' in value and '</think>' in value:
                    thinking_part = value.split('<think>')[1].split('</think>')[0].strip()
                    content_part = value.split('</think>')[1].strip()
                    # Remove <answer> tags if present
                    if '<answer>' in content_part and '</answer>' in content_part:
                        content_part = content_part.split('<answer>')[1].split('</answer>')[0].strip()
                    updated['thinking'] = thinking_part
                    updated['content'] = content_part
            
            new_messages.append(updated)
        return {'messages': new_messages}
    return {'messages': conversations if conversations else []}



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

    # Apply transformation based on type
    if conversion_type == 'type1':
        print("Applying type1 conversion...")
        dataset = dataset.map(transform_type1, desc="Converting messages to thinking format (type1)")
    elif conversion_type == 'type2':
        print("Applying type2 conversion...")
        dataset = dataset.map(transform_type2, remove_columns=['conversations'],
        desc="Converting conversations to messages with thinking format (type2)")
    else:
        print(f"Error: Unknown conversion type '{conversion_type}'")
        print("Available types: type1, type2")
        sys.exit(1)

    # Filter out invalid data after transformation
    print("Filtering valid data...")
    dataset = dataset.filter(filter_valid_data, desc="Filtering valid messages")
    print(f"After filtering: {len(dataset)}")

    # Push to hub
    print(f"Pushing to hub: {target_dataset}")
    dataset.push_to_hub(target_dataset, private=True)
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Check mode: python check-SFT-dataset.py <dataset_name>
        dataset_name = sys.argv[1]
        check_mode(dataset_name)
    elif len(sys.argv) == 4:
        # Convert mode: python check-SFT-dataset.py <source_dataset> <target_dataset> <conversion_type>
        source_dataset = sys.argv[1]
        target_dataset = sys.argv[2]
        conversion_type = sys.argv[3]
        convert_mode(source_dataset, target_dataset, conversion_type)
    else:
        print("Usage:")
        print("  Check mode:   python check-SFT-dataset.py <dataset_name>")
        print("  Convert mode: python check-SFT-dataset.py <source_dataset> <target_dataset> <conversion_type>")
        sys.exit(1)
