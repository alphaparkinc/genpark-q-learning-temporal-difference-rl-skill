"""
Autonomous Agent Tabular Q-Learning Skill
Pure Python Standard Library implementation.
"""
import random
from typing import List, Dict, Any

class QLearningAgent:
    """
    Tabular Q-Learning agent with epsilon-greedy policy and Bellman optimality updates.
    """
    def __init__(self, states: List[str], actions: List[str], alpha: float = 0.1, 
                 gamma: float = 0.9, epsilon: float = 0.1, seed: int = 42):
        self.states = list(states)
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {s: {a: 0.0 for a in actions} for s in states}
        self.rng = random.Random(seed)

    def select_action(self, state: str) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return max(self.actions, key=lambda a: self.q_table[state][a])

    def update(self, state: str, action: str, reward: float, next_state: str):
        max_next_q = max(self.q_table[next_state].values()) if next_state in self.q_table else 0.0
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def get_policy(self) -> Dict[str, str]:
        return {s: max(self.actions, key=lambda a: self.q_table[s][a]) for s in self.states}
