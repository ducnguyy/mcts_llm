import math
import numpy as np

import llm_client


class UCTNode:

    def __init__(self, answer, parent=None):
        self.answer = answer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_value = 0.0  # sum of rewards (standard UCT)

    @property
    def q_value(self):
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


def _ucb1(node, parent_visits, c=1.4):
    if node.visits == 0:
        return float('inf')
    return node.q_value + c * math.sqrt(math.log(parent_visits) / node.visits)


def _evaluate(question, answer):
    return llm_client.self_evaluate(question, answer)


def uct_answer(prompt, question, task, instance, n_simulations=16, max_children=3, c=1.4):
    llm_client.reset_call_count()
    all_nodes = []

    # Initialize root
    root_answer = llm_client.generate(prompt)
    root = UCTNode(root_answer)
    root_reward = _evaluate(question, root_answer)
    root.visits += 1
    root.total_value += root_reward
    all_nodes.append(root)

    for _ in range(n_simulations):
        node = root
        while node.children and len(node.children) >= max_children:
            node = max(node.children, key=lambda ch: _ucb1(ch, node.visits, c))

        feedback = llm_client.get_feedback(question, node.answer)
        refined = llm_client.refine(question, node.answer, feedback)

        reward = _evaluate(question, refined)

        child = UCTNode(refined, parent=node)
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
        "algorithm": "UCT",
        "iterations": n_simulations,
        "total_nodes": len(all_nodes),
        "llm_calls": llm_client.get_call_count(),
        "best_q_value": best_node.q_value,
    }
