import math
import llm_client


class PUCTNode:
    def __init__(self, answer, prior=0.5, parent=None):
        self.answer = answer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        self.prior = prior

    @property
    def q_value(self):
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


def _puct_score(node, parent_visits, c_puct=2.0):
    q = node.q_value
    u = c_puct * node.prior * math.sqrt(parent_visits) / (1 + node.visits)
    return q + u


def _evaluate(question, answer):
    return llm_client.self_evaluate(question, answer)


def puct_answer(prompt, question, task, instance, n_simulations=16, max_children=3, c_puct=2.0):
    llm_client.reset_call_count()
    all_nodes = []

    # Initialize root
    root_answer = llm_client.generate(prompt)
    root_eval = _evaluate(question, root_answer)
    root_prior = llm_client.normalize_score(root_eval)  # normalize
    root = PUCTNode(root_answer, prior=root_prior)
    root.visits = 1
    root.total_value = root_eval
    all_nodes.append(root)

    for _ in range(n_simulations):
        node = root
        while node.children and len(node.children) >= max_children:
            node = max(node.children,
                       key=lambda ch: _puct_score(ch, node.visits, c_puct))

        feedback = llm_client.get_feedback(question, node.answer)
        refined = llm_client.refine(question, node.answer, feedback)

        reward = _evaluate(question, refined)
        prior = llm_client.normalize_score(reward)

        child = PUCTNode(refined, prior=prior, parent=node)
        child.visits = 1
        child.total_value = reward
        node.children.append(child)
        all_nodes.append(child)

        current = node
        while current is not None:
            current.visits += 1
            current.total_value += reward
            current = current.parent

    best_node = max(all_nodes, key=lambda n: n.q_value)
    best_ext_reward = task.external_reward(instance, best_node.answer)

    return best_node.answer, best_ext_reward, {
        "algorithm": "PUCT",
        "iterations": n_simulations,
        "total_nodes": len(all_nodes),
        "llm_calls": llm_client.get_call_count(),
        "best_q_value": best_node.q_value,
    }
