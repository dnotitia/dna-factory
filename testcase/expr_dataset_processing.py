from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig, apply_chat_template
from datasets import load_dataset
import multiprocessing

# Load and inspect dataset structure (only first 10 samples for quick inspection)
dataset = load_dataset("cerebras/Synth-Long-SFT32K", split='train_convqa_raft')


# Function to preprocess the dataset - convert 'conversations' to 'messages'
def preprocess_dataset(example):
    """Convert conversations format to messages format for training"""
    if 'conversations' in example:
        # Rename conversations to messages
        example['messages'] = example.pop('conversations')
    return example


# Apply preprocessing to small sample first
dataset = dataset.map(preprocess_dataset, num_proc=multiprocessing.cpu_count())

# Create trainer with processed dataset
trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        dataset_num_proc=multiprocessing.cpu_count()
    ),
)

trainer.train()
