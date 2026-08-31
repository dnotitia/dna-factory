from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer

ds = Dataset.from_list(
    [
        {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]}
        for _ in range(10)
    ]
)

# Load tokenizer and set up chat template with generation keyword
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Set a chat template that supports assistant masking with {% generation %} keyword
chat_template = """{% for message in messages %}
{%- if message['role'] == 'user' -%}
{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
{%- elif message['role'] == 'assistant' -%}
{{ '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|im_end|>\n' }}
{%- endif -%}
{% endfor %}"""

tokenizer.chat_template = chat_template

args = SFTConfig(
    output_dir="./test",
    max_length=64,
    assistant_only_loss=True,
)
trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=args,
    train_dataset=ds,
    processing_class=tokenizer,
)
trainer.train()