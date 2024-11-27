# import gymnasium as gym
# import torch
# import numpy as np
# import torch.nn as nn

# # Actor Network
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, action_dim),
#             nn.Tanh()
#         )
#         self.max_action = max_action

#     def forward(self, x):
#         return self.max_action * self.network(x)

# # Critic Network
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, state, action):
#         return self.network(torch.cat([state, action], dim=1))

# # Evaluation function
# def evaluate_model(actor_model_path, critic_model_path, episodes=5):
#     env = gym.make("BipedalWalker-v3", render_mode="human")
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     max_action = float(env.action_space.high[0])

#     # Load the actor and critic models
#     actor = Actor(state_dim, action_dim, max_action)
#     actor.load_state_dict(torch.load(actor_model_path))
#     actor.eval()

#     critic = Critic(state_dim, action_dim)
#     critic.load_state_dict(torch.load(critic_model_path))
#     critic.eval()

#     for episode in range(episodes):
#         state, _ = env.reset(seed=42)
#         episode_reward = 0

#         while True:
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             action = actor(state_tensor).detach().numpy()[0]

#             # Optional: Compute Q-value of the state-action pair
#             with torch.no_grad():
#                 action_tensor = torch.FloatTensor(action).unsqueeze(0)
#                 q_value = critic(state_tensor, action_tensor).item()
#                 print(f"Q-value: {q_value:.2f}")

#             next_state, reward, terminated, truncated, _ = env.step(action)
#             state = next_state
#             episode_reward += reward

#             if terminated or truncated:
#                 print(f"Episode {episode + 1} ended with reward: {episode_reward}")
#                 break

#     env.close()

# if __name__ == "__main__":
#     # Specify the paths to the saved actor and critic models
#     actor_model_path = "models_250/ddpg_actor_episode_987_reward_286.9635572368483.pth"  # Update with your model path
#     critic_model_path = "models_250/ddpg_critic_episode_987_reward_286.9635572368483.pth"  # Update with your model path
#     evaluate_model(actor_model_path, critic_model_path)








import gymnasium as gym
import torch
import numpy as np
import os
import torch.nn as nn

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.network(x)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

# Evaluation function
def evaluate_models(models_dir, episodes=1):
    env = gym.make("BipedalWalker-v3", render_mode="human" )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Gather all actor-critic pairs
    actor_files = sorted([f for f in os.listdir(models_dir) if f.startswith("ddpg_actor")])
    critic_files = sorted([f for f in os.listdir(models_dir) if f.startswith("ddpg_critic")])

    # Ensure the pairs match
    assert len(actor_files) == len(critic_files), "Mismatch in actor and critic model counts"

    # Sort actor-critic pairs by the episode number
    pairs = sorted(zip(actor_files, critic_files), key=lambda pair: int(pair[0].split("_episode_")[1].split(".pth")[0]))

    for actor_file, critic_file in pairs:
        print(f"Evaluating Actor: {actor_file} | Critic: {critic_file}")

        # Load the actor and critic models
        actor = Actor(state_dim, action_dim, max_action)
        actor.load_state_dict(torch.load(os.path.join(models_dir, actor_file)))
        actor.eval()

        critic = Critic(state_dim, action_dim)
        critic.load_state_dict(torch.load(os.path.join(models_dir, critic_file)))
        critic.eval()

        for episode in range(episodes):
            state, _ = env.reset(seed=42)
            episode_reward = 0

            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = actor(state_tensor).detach().numpy()[0]

                # Optional: Evaluate Q-value
                with torch.no_grad():
                    action_tensor = torch.FloatTensor(action).unsqueeze(0)
                    q_value = critic(state_tensor, action_tensor).item()
                    # print(f"Q-value: {q_value:.2f}")

                next_state, reward, terminated, truncated, _ = env.step(action)
                state = next_state
                episode_reward += reward

                if terminated or truncated:
                    print(f"Episode reward: {episode_reward}")
                    break

    env.close()

if __name__ == "__main__":
    models_dir = "new_models"  # Update with your folder path if needed
    evaluate_models(models_dir)



















# import gymnasium as gym
# import torch
# import numpy as np
# import os
# import torch.nn as nn

# # Actor Network
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, action_dim),
#             nn.Tanh()
#         )
#         self.max_action = max_action

#     def forward(self, x):
#         return self.max_action * self.network(x)

# # Critic Network
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, state, action):
#         return self.network(torch.cat([state, action], dim=1))

# # Evaluation function
# def evaluate_models_by_reward(models_dir, episodes=1):
#     env = gym.make("BipedalWalker-v3", render_mode="human")
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     max_action = float(env.action_space.high[0])

#     # Gather all actor-critic pairs with rewards extracted
#     model_pairs = []
#     for filename in os.listdir(models_dir):
#         if filename.startswith("ddpg_actor"):
#             reward = float(filename.split("_reward_")[1].split(".pth")[0])
#             actor_path = os.path.join(models_dir, filename)
#             critic_path = os.path.join(models_dir, filename.replace("actor", "critic"))
#             model_pairs.append((reward, actor_path, critic_path))

#     # Sort the pairs by ascending reward
#     model_pairs.sort(key=lambda x: x[0])

#     for reward, actor_path, critic_path in model_pairs:
#         print(f"Evaluating models with Reward: {reward}")

#         # Load the actor and critic models
#         actor = Actor(state_dim, action_dim, max_action)
#         actor.load_state_dict(torch.load(actor_path))
#         actor.eval()

#         critic = Critic(state_dim, action_dim)
#         critic.load_state_dict(torch.load(critic_path))
#         critic.eval()

#         for episode in range(episodes):
#             state, _ = env.reset(seed=42)
#             episode_reward = 0

#             while True:
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0)
#                 action = actor(state_tensor).detach().numpy()[0]

#                 # Optional: Evaluate Q-value
#                 with torch.no_grad():
#                     action_tensor = torch.FloatTensor(action).unsqueeze(0)
#                     q_value = critic(state_tensor, action_tensor).item()
#                     # print(f"Q-value: {q_value:.2f}")

#                 next_state, reward_step, terminated, truncated, _ = env.step(action)
#                 state = next_state
#                 episode_reward += reward_step

#                 if terminated or truncated:
#                     print(f"Episode reward: {episode_reward}")
#                     break

#     env.close()

# if __name__ == "__main__":
#     models_dir = "final_models"  # Update with your folder path if needed
#     evaluate_models_by_reward(models_dir)
