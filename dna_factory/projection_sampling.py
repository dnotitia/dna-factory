"""Offline projection sampling, following aakaran/finetuning-with-sampling.

Accept suffix replacements strictly improving mean unconditional token logprob.
Despite the upstream name `mcmc_steps`, this is greedy acceptance, not MH.
The backend is injected so the search can be tested without a GPU.
"""

import math
import random

PROPOSAL_TEMPLATE = """You are an expert given a problem, its solution, and an initial, partial response. Carefully study the solution,
identifying what reasoning or steps are already provided, and then continue the partial response. Ensure your response is logically consistent with the solution and leads to a complete and correct final answer.

Task:

{PROBLEM}

This is an example for a solution to the problem:

{SOLUTION}

Here is a partial response:

{RESPONSE}

Starting with the partial response, continue the response in your own words, including the thinking process. Ensure the final answer exactly matches that of the provided solution.
"""


def projection_sample(
    backend,
    question,
    solution,
    *,
    max_new_tokens=512,
    block_num=16,
    mcmc_steps=10,
    rng=None,
):
    if max_new_tokens <= 0 or block_num <= 0 or max_new_tokens % block_num:
        raise ValueError("max_new_tokens must be positive and divisible by block_num")
    if mcmc_steps < 0:
        raise ValueError("mcmc_steps must be nonnegative")
    rng = rng if rng is not None else random.Random(42)
    tokens = []
    score = -math.inf
    accepted = attempts = 0
    for _ in range(block_num):
        tokens = backend.propose(
            question, solution, tokens, max_new_tokens // block_num
        )
        if not tokens:
            raise ValueError("Sampler returned an empty response")
        score = backend.score(question, tokens)
        for _ in range(mcmc_steps):
            attempts += 1
            cut = rng.randrange(len(tokens))
            candidate = backend.propose(
                question, solution, tokens[:cut], len(tokens) - cut
            )
            candidate_score = backend.score(question, candidate)
            if candidate_score > score:
                tokens, score = candidate, candidate_score
                accepted += 1
        # Like upstream, check EOS after the refinement loop, not before it.
        if backend.eos_token_id in tokens:
            tokens = tokens[: tokens.index(backend.eos_token_id) + 1]
            break
    return tokens, {"mean_logprob": score, "accepted": accepted, "attempts": attempts}


class VLLMProjectionBackend:
    def __init__(
        self,
        llm,
        *,
        temperature=0.6,
        repetition_penalty=1.1,
        template=PROPOSAL_TEMPLATE,
    ):
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        self.eos_token_id = self.tokenizer.eos_token_id
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.template = template
        if self.eos_token_id is None:
            raise ValueError("Projection sampling requires an EOS token")

    def prompt_ids(self, text):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )

    def _generate(self, ids, **kwargs):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        # Reserve one token for scoring requests as well as for generation.
        if (
            len(ids) + kwargs["max_tokens"]
            > self.llm.llm_engine.model_config.max_model_len
        ):
            raise ValueError(
                "Sampling context exceeds max_model_len; increase it or shorten the input"
            )
        return self.llm.generate(
            TokensPrompt(prompt_token_ids=ids), SamplingParams(**kwargs), use_tqdm=False
        )[0]

    def propose(self, question, solution, prefix, length):
        prompt = self.template.format(
            PROBLEM=question,
            SOLUTION=solution,
            RESPONSE=self.tokenizer.decode(prefix, skip_special_tokens=True),
        )
        output = self._generate(
            self.prompt_ids(prompt) + prefix,
            max_tokens=length,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=[self.eos_token_id],
        )
        return prefix + list(output.outputs[0].token_ids)

    def score(self, question, tokens):
        if not tokens:
            return -math.inf
        context = self.prompt_ids(question)
        result = self._generate(
            context + tokens, max_tokens=1, prompt_logprobs=0, temperature=0.0
        )
        probs = result.prompt_logprobs[len(context) :]
        if len(probs) != len(tokens) or any(
            p is None or t not in p for t, p in zip(tokens, probs)
        ):
            raise ValueError("Missing response token logprobs from vLLM")
        score = sum(p[t].logprob for t, p in zip(tokens, probs)) / len(tokens)
        if not math.isfinite(score):
            raise ValueError("Non-finite response logprob from vLLM")
        return score


def comparison_rows(question, solution, response):
    """Keep identical student prompts and row counts; never filter by correctness."""
    prompt = [{"role": "user", "content": question}]
    return {
        "sft": {"messages": prompt + [{"role": "assistant", "content": solution}]},
        "sampling_sft": {
            "messages": prompt + [{"role": "assistant", "content": response}]
        },
        "sdft": {"prompt": prompt, "privileged_context": solution},
    }
