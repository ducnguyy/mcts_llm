import math
import copy
import random
import numpy as np

import llm_client

class MCTSrNode:

    def __init__(self, answer, parent=None):
        self.answer = answer          # complete answer text
        self.parent = parent          # parent MCTSrNode or None
        self.children = []            # list of MCTSrNode
        self.rewards = []             # list of sampled reward values
        self.q_value = 0.0            # Q(a) = (min(R) + mean(R)) / 2
        self.visits = 0               # N(a)
        self.uct_value = float('inf') # UCT score

    def add_reward(self, reward):
        self.rewards.append(reward)
        self.visits = len(self.rewards)
        self.q_value = (min(self.rewards) + np.mean(self.rewards)) / 2.0

    def is_fully_expanded(self, max_children=3):
        return len(self.children) >= max_children


def _evaluate_answer(question, answer):
    return llm_client.self_evaluate(question, answer)


def _filter_mature_nodes(nodes, max_children=3):
    filtered = []
    for node in nodes:
        if not node.is_fully_expanded(max_children):
            filtered.append(node)
        else:
            # Check if any child is worse
            child_qs = [c.q_value for c in node.children]
            if max(child_qs) < node.q_value:
                filtered.append(node)
    return filtered if filtered else nodes


def _update_uct(all_nodes, c=1.4):
    epsilon = 1e-5
    for node in all_nodes:
        if node.parent is None:
            parent_visits = sum(n.visits for n in all_nodes)
        else:
            parent_visits = node.parent.visits

        node.uct_value = node.q_value + c * math.sqrt(
            math.log(parent_visits + 1) / (node.visits + epsilon)
        )


def _backpropagate(node):
    current = node.parent
    while current is not None:
        if current.children:
            max_child_q = max(c.q_value for c in current.children)
            current.q_value = (current.q_value + max_child_q) / 2.0
        current = current.parent


def mctsr(prompt, question, task, instance, max_iter=16, max_children=3, c=1.4):
    llm_client.reset_call_count()
    all_nodes = []

    # Initialize
    root_answer = llm_client.generate(prompt)
    root = MCTSrNode(root_answer)
    root_reward = _evaluate_answer(question, root_answer)
    root.add_reward(root_reward)
    all_nodes.append(root)

    # Add "I don't know" baseline node
    bad_answer = random.choice([
        "I don't know.",
        "I can't answer this question.",
        "I'm not sure how to solve this.",
    ])
    bad_node = MCTSrNode(bad_answer)
    bad_reward = _evaluate_answer(question, bad_answer)
    bad_node.add_reward(bad_reward)
    all_nodes.append(bad_node)

    # Initial UCT update
    _update_uct(all_nodes, c)

    # Main loop 
    for i in range(max_iter):
        candidates = _filter_mature_nodes(all_nodes, max_children)
        selected = max(candidates, key=lambda n: n.uct_value)
        resample_reward = _evaluate_answer(question, selected.answer)
        selected.add_reward(resample_reward)
        feedback = llm_client.get_feedback(question, selected.answer)
        refined_answer = llm_client.refine(question, selected.answer, feedback)
        child_reward = _evaluate_answer(question, refined_answer)

        child = MCTSrNode(refined_answer, parent=selected)
        child.add_reward(child_reward)
        selected.children.append(child)
        all_nodes.append(child)

        _backpropagate(child)

        _update_uct(all_nodes, c)

    best_node = max(all_nodes, key=lambda n: n.q_value)
    best_ext_reward = task.external_reward(instance, best_node.answer)

    return best_node.answer, best_ext_reward, {
        "algorithm": "MCTSr",
        "iterations": max_iter,
        "total_nodes": len(all_nodes),
        "llm_calls": llm_client.get_call_count(),
        "best_q_value": best_node.q_value,
    }
