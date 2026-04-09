import llm_client


def _evaluate(question, answer):
    return llm_client.self_evaluate(question, answer)


def single_shot(prompt, question, task, instance):
    llm_client.reset_call_count()
    answer = llm_client.generate(prompt)
    ext_reward = task.external_reward(instance, answer)

    return answer, ext_reward, {
        "algorithm": "Single-shot",
        "llm_calls": llm_client.get_call_count(),
    }

def best_of_n(prompt, question, task, instance, n=8):
    llm_client.reset_call_count()

    best_answer = None
    best_score = float("-inf")

    for _ in range(n):
        answer = llm_client.generate(prompt, temperature=0.8)
        score = _evaluate(question, answer)
        if score > best_score:
            best_score = score
            best_answer = answer

    ext_reward = task.external_reward(instance, best_answer)
    return best_answer, ext_reward, {
        "algorithm": f"Best-of-{n}",
        "n": n,
        "llm_calls": llm_client.get_call_count(),
    }


def self_refine_chain(prompt, question, task, instance, k=8):
    llm_client.reset_call_count()

    # Generate initial answer
    current_answer = llm_client.generate(prompt)
    best_answer = current_answer
    best_score = _evaluate(question, current_answer)

    # Iterative refinement
    for i in range(k):
        feedback = llm_client.get_feedback(question, current_answer)
        refined = llm_client.refine(question, current_answer, feedback)

        score = _evaluate(question, refined)
        if score >= best_score:
            best_score = score
            best_answer = refined

        current_answer = refined 

    ext_reward = task.external_reward(instance, best_answer)
    return best_answer, ext_reward, {
        "algorithm": f"SelfRefine-{k}",
        "k": k,
        "llm_calls": llm_client.get_call_count(),
    }
