import gym
import numpy as np

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Initialize Q-table
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 1000

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy policy for action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            action = np.argmax(q_table[state])  # Exploitation
        
        # Take action and observe next state and reward
        next_state, reward, done, _ = env.step(action)
        
        # Update Q-value using Q-learning equation
        q_table[state, action] += learning_rate * (reward + 
                            discount_factor * np.max(q_table[next_state]) - q_table[state, action])
        
        state = next_state
    
    # Print episode information
    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward = {reward}")

# Test the learned policy
total_rewards = []
for _ in range(100):
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward
    
    total_rewards.append(episode_reward)

print(f"Average Total Reward over 100 test episodes: {np.mean(total_rewards)}")
