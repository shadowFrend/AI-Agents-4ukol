# AI Agent - Q-Learning Grid Navigation

A reinforcement learning project implementing a Q-learning agent that learns to navigate a grid environment with obstacles to reach a goal position.

## Overview

This project implements a Q-learning agent that learns to navigate through a grid world environment. The agent uses exploration-exploitation strategy (epsilon-greedy) to learn optimal paths while avoiding obstacles.

## Features

- **Q-Learning Agent**: Implements tabular Q-learning with customizable hyperparameters
- **Grid Environment**: Configurable grid size with randomly placed obstacles
- **Web Interface**: Interactive Flask-based UI for training and visualization
- **Real-time Training**: Watch the agent learn and improve over episodes
- **Performance Metrics**: Track rewards, steps, and learning progress
- **Docker Support**: Easy deployment with Docker Compose

## Project Structure

```
.
├── app.py              # Flask web application
├── agent.py            # Q-learning agent implementation
├── environment.py      # Grid environment
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # Docker configuration
├── Dockerfile         # Docker image definition
├── templates/         # HTML templates
│   └── index.html
└── static/           # CSS and JavaScript
    ├── style.css
    └── script.js
```

## Installation

### Local Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

### Docker Setup

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. Access the application at `http://localhost:5000`

## How It Works

### Q-Learning Agent

The agent uses Q-learning algorithm with:
- **Learning Rate**: 0.1
- **Discount Factor**: 0.95
- **Epsilon**: 1.0 (decays to 0.01)
- **Epsilon Decay**: 0.995

### Grid Environment

- **Actions**: 4 directions (up, right, down, left)
- **Start Position**: Top-left corner (0, 0)
- **Goal Position**: Bottom-right corner
- **Obstacles**: 20% of grid cells (randomly placed)

### Reward Structure

- **Goal Reached**: +100
- **Each Step**: -1
- **Invalid Move** (wall/obstacle): -5

## API Endpoints

- `POST /initialize` - Initialize environment with specified grid size
- `POST /train` - Train agent for specified number of episodes
- `POST /evaluate` - Evaluate agent's learned policy
- `GET /stats` - Get training statistics

## Usage

1. **Initialize Environment**: Set the grid size and initialize the environment
2. **Train Agent**: Specify number of episodes and start training
3. **Evaluate**: Test the trained agent's performance with greedy policy
4. **Visualize**: View training progress and agent's learned paths

## Configuration

Hyperparameters can be adjusted in `app.py`:
- `grid_size`: Size of the grid (default: 5)
- `learning_rate`: Agent's learning rate (default: 0.1)
- `discount_factor`: Future reward discount (default: 0.95)
- `epsilon`: Initial exploration rate (default: 1.0)
- `epsilon_decay`: Exploration decay rate (default: 0.995)

## License

This project is available for educational purposes.
