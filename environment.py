import numpy as np
import random

class GridEnvironment:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size

        # Define action space: 0=up, 1=right, 2=down, 3=left
        self.n_actions = 4
        self.actions = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1)   # left
        }

        # Initialize grid
        self.reset_environment()

    def reset_environment(self):
        """Initialize start, goal, and obstacles"""
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Set start position (top-left corner)
        self.start_pos = [0, 0]

        # Set goal position (bottom-right corner)
        self.goal_pos = [self.grid_size-1, self.grid_size-1]

        # Create obstacles (20% of cells, excluding start and goal)
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        n_obstacles = max(1, int(0.2 * self.grid_size * self.grid_size))

        # Get positions adjacent to goal
        goal_neighbors = []
        for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_pos = [self.goal_pos[0] + delta[0], self.goal_pos[1] + delta[1]]
            if 0 <= neighbor_pos[0] < self.grid_size and 0 <= neighbor_pos[1] < self.grid_size:
                goal_neighbors.append(neighbor_pos)

        obstacle_positions = []
        while len(obstacle_positions) < n_obstacles:
            pos = [random.randint(0, self.grid_size-1),
                   random.randint(0, self.grid_size-1)]
            # Exclude start, goal, and positions adjacent to goal
            if (pos != self.start_pos and pos != self.goal_pos and
                pos not in goal_neighbors):
                if pos not in obstacle_positions:
                    obstacle_positions.append(pos)
                    self.obstacles[pos[0], pos[1]] = True

        self.agent_pos = self.start_pos.copy()

    def reset(self):
        """Reset agent to start position"""
        self.agent_pos = self.start_pos.copy()
        return self.pos_to_state(self.agent_pos)

    def pos_to_state(self, pos):
        """Convert grid position to state number"""
        return pos[0] * self.grid_size + pos[1]

    def state_to_pos(self, state):
        """Convert state number to grid position"""
        return [state // self.grid_size, state % self.grid_size]

    def is_valid_position(self, pos):
        """Check if position is valid (within bounds and not obstacle)"""
        if pos[0] < 0 or pos[0] >= self.grid_size:
            return False
        if pos[1] < 0 or pos[1] >= self.grid_size:
            return False
        if self.obstacles[pos[0], pos[1]]:
            return False
        return True

    def step(self, action):
        """Execute action and return new state, reward, done"""
        # Calculate new position
        action_delta = self.actions[action]
        new_pos = [
            self.agent_pos[0] + action_delta[0],
            self.agent_pos[1] + action_delta[1]
        ]

        # Check if new position is valid
        if self.is_valid_position(new_pos):
            self.agent_pos = new_pos

            # Check if goal reached
            if self.agent_pos == self.goal_pos:
                reward = 100  # Large positive reward for reaching goal
                done = True
            else:
                reward = -1  # Small negative reward for each step
                done = False
        else:
            # Invalid move (hit wall or obstacle)
            reward = -5  # Penalty for hitting wall/obstacle
            done = False

        return self.pos_to_state(self.agent_pos), reward, done
