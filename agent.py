import numpy as np
import random

class QTableAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with small random values
        self.q_table = np.random.uniform(low=-0.01, high=0.01,
                                         size=(n_states, n_actions))

    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            # Explore: choose random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploit: choose best action from Q-table
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning algorithm"""
        current_q = self.q_table[state, action]

        if done:
            target_q = reward
        else:
            # Q-learning: use max Q-value of next state
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])

        # Update Q-value
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """Get greedy policy from Q-table"""
        return np.argmax(self.q_table, axis=1)
