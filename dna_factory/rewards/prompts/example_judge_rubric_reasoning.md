You are a strict grader for a reasoning task. You are given a QUESTION, a REFERENCE ANSWER (the
known-correct answer), and a model's RESPONSE.

Work step by step: identify the RESPONSE's final answer and compare it to the REFERENCE ANSWER for
mathematical / logical / factual equivalence. The reasoning, method, or wording may differ from the
reference — judge ONLY whether the RESPONSE's final answer is correct with respect to the REFERENCE
ANSWER, not its style or how closely it mirrors the reference's derivation.

Rate the RESPONSE from 0 to 10:
0-2  = wrong: the final answer contradicts the reference, or no final answer is given
3-4  = mostly wrong: right idea but the final answer is incorrect
5-6  = partially correct: on the right track, but the final answer is incomplete or only partly right
7-8  = correct: the final answer matches the reference, with at most minor slips in rigor/justification
9-10 = fully correct: the final answer matches the reference and the solution is sound and complete

Equivalent forms count as matching (e.g. 1/2 = 0.5, simplified vs unsimplified, reordered sets,
algebraically equivalent expressions, different but valid phrasing of the same fact). Do NOT give
credit to a response that hedges among several answers without committing to the correct one.

QUESTION:
{prompt}

REFERENCE ANSWER:
{reference}

RESPONSE:
{completion}

Briefly justify your grade in 1-3 sentences, then output the final line in exactly this format,
where N is an integer from 0 to 10:
Score: N
