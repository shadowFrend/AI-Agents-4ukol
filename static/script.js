let rewardChart = null;
let gridSize = 5;
let environment = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeChart();

    document.getElementById('initBtn').addEventListener('click', initializeEnvironment);
    document.getElementById('trainBtn').addEventListener('click', trainAgent);
    document.getElementById('evaluateBtn').addEventListener('click', evaluateAgent);
});

function initializeChart() {
    const ctx = document.getElementById('rewardChart').getContext('2d');
    rewardChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Episode Reward',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Reward'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Episode'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        }
    });
}

async function initializeEnvironment() {
    try {
        gridSize = parseInt(document.getElementById('gridSize').value);

        const response = await fetch('/initialize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ grid_size: gridSize })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            environment = data;
            renderGrid(data);
            renderQValuesGrid(gridSize);

            document.getElementById('trainBtn').disabled = false;
            document.getElementById('evaluateBtn').disabled = true;

            // Reset chart
            rewardChart.data.labels = [];
            rewardChart.data.datasets[0].data = [];
            rewardChart.update();

            // Reset stats
            document.getElementById('totalEpisodes').textContent = '0';
            document.getElementById('currentEpsilon').textContent = '1.00';
            document.getElementById('lastReward').textContent = '-';
            document.getElementById('goalReached').textContent = '-';
        } else {
            alert('Failed to initialize environment: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error initializing environment:', error);
        alert('Error initializing environment. Please try again.');
    }
}

function renderGrid(envData, path = null) {
    const gridContainer = document.getElementById('grid');
    gridContainer.innerHTML = '';

    for (let i = 0; i < gridSize; i++) {
        const row = document.createElement('div');
        row.className = 'grid-row';

        for (let j = 0; j < gridSize; j++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';

            // Check cell type
            if (i === envData.start_pos[0] && j === envData.start_pos[1]) {
                cell.classList.add('start');
                cell.textContent = 'S';
            } else if (i === envData.goal_pos[0] && j === envData.goal_pos[1]) {
                cell.classList.add('goal');
                cell.textContent = 'G';
            } else if (envData.obstacles[i][j]) {
                cell.classList.add('obstacle');
            }

            // Add path if provided
            if (path) {
                for (let p of path) {
                    if (p[0] === i && p[1] === j) {
                        if (!cell.classList.contains('start') && !cell.classList.contains('goal')) {
                            cell.classList.add('path');
                        }
                    }
                }

                // Mark current agent position
                const lastPos = path[path.length - 1];
                if (lastPos[0] === i && lastPos[1] === j &&
                    !(i === envData.goal_pos[0] && j === envData.goal_pos[1])) {
                    cell.classList.add('agent');
                    cell.textContent = '🤖';
                }
            }

            row.appendChild(cell);
        }
        gridContainer.appendChild(row);
    }
}

function renderQValuesGrid(size, qTable = null) {
    const gridContainer = document.getElementById('qValuesGrid');
    gridContainer.innerHTML = '';

    for (let i = 0; i < size; i++) {
        const row = document.createElement('div');
        row.className = 'grid-row';

        for (let j = 0; j < size; j++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell q-value-cell';

            if (qTable) {
                const stateIdx = i * size + j;
                const qValues = qTable[stateIdx];
                const maxQ = Math.max(...qValues);
                const bestAction = qValues.indexOf(maxQ);

                // Color based on max Q-value
                const intensity = Math.min(255, Math.max(0, Math.floor((maxQ + 10) * 10)));
                cell.style.backgroundColor = `rgb(${255-intensity}, ${255}, ${255-intensity})`;

                // Show best action arrow
                const arrows = ['↑', '→', '↓', '←'];
                cell.innerHTML = `<span class="best-action">${arrows[bestAction]}</span>`;
            }

            row.appendChild(cell);
        }
        gridContainer.appendChild(row);
    }
}

async function trainAgent() {
    try {
        const episodes = parseInt(document.getElementById('episodes').value);

        document.getElementById('trainBtn').disabled = true;
        document.getElementById('trainBtn').textContent = 'Training...';

        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ episodes: episodes })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            // Update chart
            const currentEpisodes = rewardChart.data.labels.length;
            for (let i = 0; i < data.episode_rewards.length; i++) {
                rewardChart.data.labels.push(currentEpisodes + i + 1);
                rewardChart.data.datasets[0].data.push(data.episode_rewards[i]);
            }
            rewardChart.update();

            // Update stats
            document.getElementById('totalEpisodes').textContent =
                rewardChart.data.labels.length;
            document.getElementById('currentEpsilon').textContent =
                data.final_epsilon.toFixed(3);
            document.getElementById('lastReward').textContent =
                data.episode_rewards[data.episode_rewards.length - 1].toFixed(1);

            // Update Q-values grid
            renderQValuesGrid(gridSize, data.q_table);

            // Show last path
            if (data.paths.length > 0) {
                const lastPath = data.paths[data.paths.length - 1];
                renderGrid(environment, lastPath);
            }

            document.getElementById('evaluateBtn').disabled = false;
        } else {
            alert('Training failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error training agent:', error);
        alert('Error training agent. Please try again.');
    } finally {
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('trainBtn').textContent = 'Train Agent';
    }
}

async function evaluateAgent() {
    try {
        const response = await fetch('/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            // Animate path
            animatePath(data.path);

            // Update stats
            document.getElementById('lastReward').textContent =
                data.total_reward.toFixed(1);
            document.getElementById('goalReached').textContent =
                data.reached_goal ? '✅ Yes' : '❌ No';
        } else {
            alert('Evaluation failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error evaluating agent:', error);
        alert('Error evaluating agent. Please try again.');
    }
}

function animatePath(path) {
    let step = 0;
    const interval = setInterval(() => {
        if (step < path.length) {
            renderGrid(environment, path.slice(0, step + 1));
            step++;
        } else {
            clearInterval(interval);
        }
    }, 200);
}
