"""Measure the on-policy reverse KL, KL(student || teacher), on a fixed held-out prompt set.

This is the quantity on-policy distillation actually minimizes: the student samples its own
completions, and we score the per-token divergence of its next-token distribution from the
teacher's at exactly those states. Run it on the base student and on the distilled checkpoint
with the same prompts/seed to see whether training moved the objective on unseen prompts.

    $ python testcase/expr_distillation_reverse_kl.py \
        --student dnotitia/Qwen3-0.6B --teacher dnotitia/Qwen3-1.7B      # before
    $ python testcase/expr_distillation_reverse_kl.py \
        --student ./my-distilled-checkpoint --tokenizer dnotitia/Qwen3-1.7B \
        --teacher dnotitia/Qwen3-1.7B                                    # after
"""
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_completion_mask(completion_ids, eos_ids):
    """1 for every generated token up to and including the first EOS, 0 afterwards."""
    is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
    for eos in eos_ids:
        is_eos |= completion_ids == eos
    # cumulative count of EOS seen strictly before this position
    eos_before = is_eos.cumsum(dim=1) - is_eos.long()
    return (eos_before == 0).long()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--tokenizer", default=None, help="defaults to --teacher")
    ap.add_argument("--dataset", default="trl-lib/ultrafeedback-prompt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.teacher)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_ids = tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id]
    if tok.pad_token_id is not None:
        eos_ids = list({*eos_ids, tok.pad_token_id})

    ds = load_dataset(args.dataset, split=args.split).select(range(args.num_prompts))
    texts = [
        tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        for ex in ds
    ]

    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    kl_sum, tok_count, len_sum = 0.0, 0, 0
    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start:start + args.batch_size]
        enc = tok(batch_texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        prompt_len = enc["input_ids"].shape[1]

        torch.manual_seed(args.seed + start)
        out = student.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=1.0,
            top_k=0,
            pad_token_id=tok.pad_token_id,
        )
        completion_ids = out[:, prompt_len:]
        completion_mask = build_completion_mask(completion_ids, eos_ids)
        attention_mask = torch.cat([enc["attention_mask"], completion_mask], dim=1)

        s_logits = student(input_ids=out, attention_mask=attention_mask).logits
        t_logits = teacher(input_ids=out, attention_mask=attention_mask).logits
        # logits at position i predict token i+1, so the completion tokens are predicted from
        # positions [prompt_len - 1, ..., end - 1]
        s_logits = s_logits[:, prompt_len - 1:-1].float() / args.temperature
        t_logits = t_logits[:, prompt_len - 1:-1].float() / args.temperature

        s_logprobs = F.log_softmax(s_logits, dim=-1)
        t_logprobs = F.log_softmax(t_logits, dim=-1)
        # reverse KL: sum_v p_student(v) * (log p_student(v) - log p_teacher(v))
        per_token_kl = (s_logprobs.exp() * (s_logprobs - t_logprobs)).sum(dim=-1)

        kl_sum += (per_token_kl * completion_mask).sum().item()
        tok_count += completion_mask.sum().item()
        len_sum += completion_mask.sum(dim=1).float().mean().item() * len(batch_texts)
        del s_logits, t_logits, s_logprobs, t_logprobs, per_token_kl
        torch.cuda.empty_cache()

    print(f"student            : {args.student}")
    print(f"teacher            : {args.teacher}")
    print(f"prompts            : {len(texts)} from {args.dataset}[{args.split}]")
    print(f"completion tokens  : {tok_count}")
    print(f"mean completion len: {len_sum / len(texts):.1f}")
    print(f"MEAN REVERSE KL    : {kl_sum / tok_count:.6f} nats/token")


if __name__ == "__main__":
    main()
