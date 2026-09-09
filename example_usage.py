"""Example usage for Q-Learning Skill."""
from client import QLearningAgent

def main():
    print("Executing Tabular Q-Learning...")
    states = ["State_0", "State_1", "Goal_State"]
    actions = ["MOVE_LEFT", "MOVE_RIGHT"]
    agent = QLearningAgent(states, actions, alpha=0.2, gamma=0.9, epsilon=0.05)

    for _ in range(100):
        agent.update("State_0", "MOVE_RIGHT", 0.0, "State_1")
        agent.update("State_1", "MOVE_RIGHT", 10.0, "Goal_State")

    policy = agent.get_policy()
    print("Learned Policy:", policy)
    assert policy["State_0"] == "MOVE_RIGHT"
    assert policy["State_1"] == "MOVE_RIGHT"
    print("Q-Learning verified successfully!")

if __name__ == "__main__":
    main()
