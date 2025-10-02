from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
from environment import GridEnvironment
from agent import QTableAgent
import numpy as np

app = Flask(__name__)
CORS(app)

# Global variables to store environment and agent
env = None
agent = None
training_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize():
    global env, agent, training_history

    try:
        data = request.json
        grid_size = data.get('grid_size', 5)

        # Validate grid size
        if not isinstance(grid_size, int) or grid_size < 3 or grid_size > 10:
            return jsonify({'success': False, 'error': 'Grid size must be between 3 and 10'}), 400

        # Initialize environment and agent
        env = GridEnvironment(grid_size=grid_size)
        agent = QTableAgent(
            n_states=grid_size * grid_size,
            n_actions=4,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        training_history = []

        return jsonify({
            'success': True,
            'grid_size': grid_size,
            'start_pos': env.start_pos,
            'goal_pos': env.goal_pos,
            'obstacles': env.obstacles.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    global env, agent, training_history

    try:
        if env is None or agent is None:
            return jsonify({'success': False, 'error': 'Environment not initialized'}), 400

        data = request.json
        episodes = data.get('episodes', 100)

        # Validate episodes
        if not isinstance(episodes, int) or episodes < 1 or episodes > 1000:
            return jsonify({'success': False, 'error': 'Episodes must be between 1 and 1000'}), 400

        episode_rewards = []
        episode_steps = []
        paths = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            episode_path = [env.agent_pos.copy()]

            done = False
            while not done and steps < 100:  # Max 100 steps per episode
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1
                episode_path.append(env.agent_pos.copy())

            episode_rewards.append(total_reward)
            episode_steps.append(steps)

            # Save path for last few episodes
            if episode >= episodes - 5:
                paths.append(episode_path)

            agent.decay_epsilon()

        training_history.extend(episode_rewards)

        return jsonify({
            'success': True,
            'episode_rewards': episode_rewards,
            'episode_steps': episode_steps,
            'final_epsilon': agent.epsilon,
            'paths': paths,
            'q_table': agent.q_table.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    global env, agent

    try:
        if env is None or agent is None:
            return jsonify({'success': False, 'error': 'Environment not initialized'}), 400

        # Run evaluation with greedy policy
        state = env.reset()
        path = [env.agent_pos.copy()]
        total_reward = 0
        steps = 0

        done = False
        while not done and steps < 100:
            action = agent.choose_action(state, training=False)  # Greedy action
            next_state, reward, done = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1
            path.append(env.agent_pos.copy())

        return jsonify({
            'success': True,
            'path': path,
            'total_reward': total_reward,
            'steps': steps,
            'reached_goal': done
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    global training_history

    return jsonify({
        'training_history': training_history,
        'total_episodes': len(training_history)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
