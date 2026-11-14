# Your first SFT run with public data

This example checks that model loading, data preparation, GPU training, and saving
work together. It uses the public [Qwen3-0.6B-Base model](https://huggingface.co/dnotitia/Qwen3-0.6B-Base)
and [Capybara conversations](https://huggingface.co/datasets/trl-lib/Capybara), with no
Hugging Face login, private dataset access, W&B account, or judge endpoint required.

The run selects 32 training conversations, limits sequences to 256 tokens, and stops
after **two optimizer steps**. It is an environment check, not a useful fine-tuned
model or a model-quality benchmark. Review the model and dataset cards before using
them beyond this example.

## 1. Set up

Use a Linux machine with an NVIDIA GPU supporting bfloat16, a compatible NVIDIA
driver, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The example was tested on one RTX 4090 with 24 GB VRAM; smaller GPUs have not been
validated. The project requires Python 3.14 or newer; uv can install the Python
version specified by the repository.

```bash
git clone https://github.com/dnotitia/dna-factory.git
cd dna-factory
uv sync --locked
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Internet access and space for the dependencies, model, dataset cache, and saved
model are required. Selecting `train[:32]` limits the training examples, but the
dataset library may still download the complete underlying data file. The first
run includes downloads and preprocessing, so it takes longer than training alone.

## 2. Run two training steps

From the repository root:

```bash
python sft.py --config configs/SFT/qwen3-0.6B-public-smoke.yaml
```

The configuration uses PyTorch SDPA with packing and Liger disabled. This avoids
an additional FlashAttention kernel download for the first run. W&B reporting,
periodic checkpoints, and checkpoint evaluation are disabled. The normal final
model save still runs.

These settings are local overrides; the shared defaults and Korean training
examples are unchanged.

## 3. Check the result

A successful run finishes both steps and saves the model and tokenizer under
`outputs/qwen3-0.6b-public-smoke/`. Check the training metrics:

```bash
python -m json.tool outputs/qwen3-0.6b-public-smoke/train_results.json
```

The output includes `train_loss` and `train_runtime`. Loss can vary between runs;
there is no target quality score for this check. You should also find
`model.safetensors`, `config.json`, and tokenizer files in the output directory.

Repeated runs use the same output directory. To keep an earlier result, choose a
different directory with a CLI override:

```bash
python sft.py --config configs/SFT/qwen3-0.6B-public-smoke.yaml \
  --output_dir outputs/my-first-sft
```

## Troubleshooting

- **CUDA is unavailable:** run `nvidia-smi` and the CUDA check above on the GPU
  machine. Installing the Python dependencies on a laptop does not provide a
  remote GPU connection.
- **Hub access or download errors:** confirm the model and dataset pages above
  are reachable. Authentication is not required for these public resources, but
  network restrictions and anonymous rate limits can still prevent a download.
- **A Korean example raises `DatasetNotFoundError`:** verify that you have access
  to that example's dataset with its owner. This public-data configuration is a
  separate way to check the training environment.
- **GPU out of memory:** confirm that another process is not using the GPU and
  that this configuration is being used. The validated setup is a 24 GB RTX 4090.

After the check, use a separate configuration for a longer training run: choose
appropriate data, sequence length, training steps, and reporting options for your
goal. If you rented a GPU, stop or release it when finished according to your
provider's billing and storage rules.
