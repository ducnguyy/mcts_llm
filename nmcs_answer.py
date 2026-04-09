import llm_client


def _evaluate(question, answer):
    return llm_client.self_evaluate(question, answer)


def _nmcs_recursive(prompt, question, answer, task, instance, level, max_refinements=3):
    # Level 0: single evaluation 
    if level == 0:
        if answer is None:
            answer = llm_client.generate(prompt)
        score = _evaluate(question, answer)
        return answer, score

    # Initialize with current answer or a fresh one
    if answer is None:
        answer = llm_client.generate(prompt)

    best_answer = answer
    best_score = _evaluate(question, answer)

    for _ in range(max_refinements):
        feedback = llm_client.get_feedback(question, answer) 
        refined = llm_client.refine(question, answer, feedback) 

        # Recursively evaluate at level-1
        result_answer, result_score = _nmcs_recursive(
            prompt, question, refined, task, instance,
            level - 1, max_refinements
        )

        if result_score >= best_score:
            best_score = result_score
            best_answer = result_answer

    return best_answer, best_score


def nmcs_answer(prompt, question, task, instance, level=1, max_refinements=3):
    llm_client.reset_call_count()

    best_answer, best_score = _nmcs_recursive(
        prompt, question, None, task, instance,
        level, max_refinements
    )

    best_ext_reward = task.external_reward(instance, best_answer)

    return best_answer, best_ext_reward, {
        "algorithm": f"NMCS-{level}",
        "level": level,
        "max_refinements": max_refinements,
        "llm_calls": llm_client.get_call_count(),
        "best_score": best_score,
    }
