import gym
import numpy as np

# Création de l'environnement
env = gym.make("FrozenLake-v1", is_slippery=True)

# Paramètres de Q-learning
learning_rate = 0.8
discount_factor = 0.95
num_episodes = 2000
exploration_rate = 1.0
exploration_decay = 0.995
min_exploration_rate = 0.01

# Initialisation de la table Q
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# Fonction pour choisir une action en fonction de la politique 
def choose_action(state):
    if np.random.rand() < exploration_rate:
        return env.action_space.sample()  # Exploration
    return np.argmax(q_table[state, :])  # Exploitation

# Boucle principale d'apprentissage
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_rewards = 0

    while not done:
        action = choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Mise à jour de la table Q
        best_next_action = np.argmax(q_table[next_state, :])
        td_target = reward + discount_factor * q_table[next_state, best_next_action]
        td_error = td_target - q_table[state, action]
        q_table[state, action] += learning_rate * td_error

        state = next_state
        total_rewards += reward

    # Mise à jour du taux d'exploration
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}: Total Rewards = {total_rewards}")

# Test du modèle appris
state, _ = env.reset()
done = False
print("\nPolicy Apprise:")
while not done:
    action = np.argmax(q_table[state, :])
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    state = next_state

env.close()
