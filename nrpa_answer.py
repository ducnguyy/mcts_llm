import math
import copy
import random

import llm_client


# Refinement strategies 
REFINEMENT_STYLES = [
    "Focus on improving the structure and format of the answer.",
    "Focus on improving content accuracy and correctness.",
    "Focus on improving creativity, quality, and expressiveness.",
    "Focus on making the answer more concise and precise.",
    "Try a completely different approach to answer this question.",
]


def _evaluate(question, answer):
    return llm_client.self_evaluate(question, answer)


def _sample_style(policy):
    max_w = max(policy)
    exps = [math.exp(w - max_w) for w in policy]
    total = sum(exps)
    r = random.random() * total
    cumsum = 0.0
    for i, e in enumerate(exps):
        cumsum += e
        if cumsum >= r:
            return i
    return len(policy) - 1


def _playout(prompt, question, task, instance, policy, current_answer=None):
    style_idx = _sample_style(policy)
    style = REFINEMENT_STYLES[style_idx]

    if current_answer is None:
        # Generate from scratch with style hint
        answer = llm_client.generate(
            prompt + f"\n\nAdditional guidance: {style}",
            temperature=0.8
        )
    else:
        # Refine
        feedback = llm_client.get_feedback(
            question,
            current_answer + f"\n\n{style}"
        )
        answer = llm_client.refine(question, current_answer, feedback)

    score = _evaluate(question, answer)
    return score, answer, style_idx


def _adapt(policy, best_style_sequence, alpha=1.0):
    polp = copy.copy(policy)

    for best_idx in best_style_sequence:
        max_w = max(policy)
        exps = [math.exp(w - max_w) for w in policy]
        z = sum(exps)

        # Update: increase best, decrease others proportionally
        for i in range(len(polp)):
            polp[i] -= alpha * exps[i] / z

        polp[best_idx] += alpha

    return polp


def _nrpa_recursive(prompt, question, task, instance, level, policy, n_iter, current_answer=None):
    if level == 0:
        return _playout(prompt, question, task, instance, policy, current_answer)

    best_score = float('-inf')
    best_answer = None
    best_styles = []

    for i in range(n_iter):
        pol = copy.copy(policy)
        score, answer, style_idx = _nrpa_recursive(
            prompt, question, task, instance,
            level - 1, pol, n_iter, current_answer
        )

        if score > best_score:
            best_score = score
            best_answer = answer
            best_styles = [style_idx]

        policy = _adapt(policy, best_styles)

    return best_score, best_answer, best_styles[0] if best_styles else 0


def nrpa_answer(prompt, question, task, instance, level=1, n_iter=8):
    llm_client.reset_call_count()

    # Initialize policy 
    policy = [0.0] * len(REFINEMENT_STYLES)

    best_score, best_answer, _ = _nrpa_recursive(
        prompt, question, task, instance,
        level, policy, n_iter
    )

    best_ext_reward = task.external_reward(instance, best_answer)

    return best_answer, best_ext_reward, {
        "algorithm": f"NRPA-{level}",
        "level": level,
        "n_iter": n_iter,
        "llm_calls": llm_client.get_call_count(),
        "best_score": best_score,
    }
