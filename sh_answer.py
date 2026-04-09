import math

import llm_client


def _evaluate(question, answer):
    return llm_client.self_evaluate(question, answer)


def sh_answer(prompt, question, task, instance, n_candidates=8, n_rounds=None):
    llm_client.reset_call_count()

    if n_rounds is None:
        n_rounds = max(1, int(math.ceil(math.log2(n_candidates))))

    candidates = []
    for _ in range(n_candidates):
        answer = llm_client.generate(prompt, temperature=0.8)
        score = _evaluate(question, answer)
        candidates.append({"answer": answer, "score": score})

    for round_idx in range(n_rounds):
        if len(candidates) <= 1:
            break

        # Sort by score and keep top half
        candidates.sort(key=lambda c: c["score"], reverse=True)
        n_survivors = max(1, len(candidates) // 2)
        candidates = candidates[:n_survivors]

        # Refine survivors and re-evaluate
        refined_candidates = []
        for cand in candidates:
            feedback = llm_client.get_feedback(question, cand["answer"])
            refined = llm_client.refine(question, cand["answer"], feedback)
            refined_score = _evaluate(question, refined)

            # Keep the better of original and refined
            if refined_score >= cand["score"]:
                refined_candidates.append({
                    "answer": refined,
                    "score": refined_score
                })
            else:
                refined_candidates.append(cand)

        candidates = refined_candidates

    best = max(candidates, key=lambda c: c["score"])
    best_ext_reward = task.external_reward(instance, best["answer"])

    return best["answer"], best_ext_reward, {
        "algorithm": "SH",
        "n_candidates": n_candidates,
        "n_rounds": n_rounds,
        "llm_calls": llm_client.get_call_count(),
        "best_score": best["score"],
    }
