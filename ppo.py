import ale_py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import distributions

from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

from network import CNNPolicy, CNNValue, ActorCritic

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


class PPO:
    def __init__(self,
                 env_name="PhoenixNoFrameskip-v4",
                 hidden_dimensions=512,
                 discount_factor=0.99,
                 gae_lambda=0.95,
                 ppo_steps=8,
                 epsilon=0.2,
                 entropy_coefficient=0.01,
                 learning_rate=2.5e-4,
                 rollout_length=128,
                 max_episodes=2000,
                 reward_threshold=2000,
                 print_interval=10,
                 n_trials=100,
                 n_envs_train=8,
                 n_envs_test=1):
    
        print(f'Using device: {device.__str__().upper()}')
        
        self.env_train = make_atari_env(env_name, n_envs=n_envs_train, seed=0)
        self.env_train = VecFrameStack(self.env_train, n_stack=4)
        
        self.env_test = make_atari_env(env_name, n_envs=n_envs_test, seed=42)
        self.env_test = VecFrameStack(self.env_test, n_stack=4)
        
        obs_shape = self.env_train.observation_space.shape  # (84, 84, 4)
        obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (4, 84, 84)
        act_dim = self.env_train.action_space.n
        
        self.agent = ActorCritic(
            CNNPolicy(obs_shape, hidden_dimensions, act_dim),
            CNNValue(obs_shape, hidden_dimensions)
        ).to(device)
        
        self.initial_lr = learning_rate
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.initial_lr)
        
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.ppo_steps = ppo_steps
        self.epsilon = epsilon
        self.entropy_coefficient = entropy_coefficient
        self.rollout_length = rollout_length * n_envs_train
        self.max_episodes = max_episodes
        self.reward_threshold = reward_threshold
        self.print_interval = print_interval
        self.n_trials = n_trials
        
        # логови
        self.train_rewards = []
        self.test_rewards = []
        self.policy_losses = []
        self.value_losses = []

    def _calculate_returns_and_advantages(self, rewards, dones, values, last_value):
        T = len(rewards)
        returns = torch.zeros(T, dtype=torch.float32, device=device)
        adv = torch.zeros(T, dtype=torch.float32, device=device)
        
        gae = 0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            next_value = last_value if t == T - 1 else values[t + 1]
            
            delta = rewards[t] + self.discount_factor * next_value * next_nonterminal - values[t]
            gae = delta + self.discount_factor * self.gae_lambda * next_nonterminal * gae
            adv[t] = gae
            returns[t] = adv[t] + values[t]
            
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + EPS)
        return returns, adv

    def _calculate_surrogate_loss(self, old_logps, new_logps, advantages):
        ratio = (new_logps - old_logps).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        return torch.min(surr1, surr2)

    def _calculate_losses(self, surrogate_loss, entropy, returns, value_pred):
        policy_loss = -surrogate_loss.mean()
        value_loss = F.smooth_l1_loss(value_pred, returns)
        entropy_loss = -self.entropy_coefficient * entropy.mean()
        return policy_loss, value_loss, entropy_loss

        
    def collect_trajectories(self):
        n_envs = self.env_train.num_envs
        
        states_per_env = [[] for _ in range(n_envs)]
        actions_per_env = [[] for _ in range(n_envs)]
        logps_per_env = [[] for _ in range(n_envs)]
        values_per_env = [[] for _ in range(n_envs)]
        rewards_per_env = [[] for _ in range(n_envs)]
        dones_per_env = [[] for _ in range(n_envs)]
        
        ep_rewards = np.zeros(n_envs, dtype=np.float32)   # тековниот исход од епизодата
        finished_episodes = []                            # исходот од тековните завршени епизоди
        
        state = self.env_train.reset()
        
        for _ in range(self.rollout_length):
            state_tensor = torch.FloatTensor(state).permute(0, 3, 1, 2).to(device)
            with torch.no_grad():
                action_logits, value_pred = self.agent(state_tensor)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = distributions.Categorical(action_probs)
                actions = dist.sample()
                logps = dist.log_prob(actions)
            
            next_state, reward, done, info = self.env_train.step(actions)
            
            for i in range(n_envs):
                states_per_env[i].append(state_tensor[i:i+1].cpu())
                actions_per_env[i].append(actions[i].unsqueeze(0).cpu())
                logps_per_env[i].append(logps[i].unsqueeze(0).cpu())
                values_per_env[i].append(value_pred[i].unsqueeze(0).cpu())
                rewards_per_env[i].append(reward[i])
                dones_per_env[i].append(done[i])
                
                # акумулирање на вкупната награда од епизодата
                ep_rewards[i] += reward[i]
                if done[i]:
                    finished_episodes.append(ep_rewards[i])
                    ep_rewards[i] = 0.0
    
            state = next_state
    
        next_state_tensor = torch.FloatTensor(next_state).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            _, next_value_pred = self.agent(next_state_tensor)
        next_values = next_value_pred.squeeze(-1)
    
        all_states, all_actions, all_logps, all_returns, all_adv = [], [], [], [], []
        for i in range(n_envs):
            env_states = torch.cat(states_per_env[i], dim=0)
            env_actions = torch.cat(actions_per_env[i], dim=0)
            env_logps = torch.cat(logps_per_env[i], dim=0)
            env_values = torch.cat(values_per_env[i], dim=0).squeeze(-1)
            env_rewards = torch.tensor(rewards_per_env[i], dtype=torch.float32, device=device)
            env_dones = torch.tensor(dones_per_env[i], dtype=torch.float32, device=device)
            
            env_returns, env_adv = self._calculate_returns_and_advantages(
                env_rewards, env_dones, env_values, next_values[i].item()
            )
    
            all_states.append(env_states)
            all_actions.append(env_actions)
            all_logps.append(env_logps)
            all_returns.append(env_returns)
            all_adv.append(env_adv)
    
        states = torch.cat(all_states, dim=0)
        actions = torch.cat(all_actions, dim=0)
        old_logps = torch.cat(all_logps, dim=0)
        returns = torch.cat(all_returns, dim=0)
        advantages = torch.cat(all_adv, dim=0)
        
        # исто како SB3: просек од последните 100 завршени епизоди
        ep_rew_mean = np.mean(finished_episodes[-100:]) if finished_episodes else 0.0
    
        return ep_rew_mean, states, actions, old_logps, advantages, returns

    def update_policy(self, states, actions, old_logps, advantages, returns, batch_size=256):
        self.agent.train()
        dataset = TensorDataset(states, actions, old_logps, advantages, returns)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_policy_loss, total_value_loss, total_entropy_loss = 0, 0, 0
        for _ in range(self.ppo_steps):
            for b_states, b_actions, b_old_logps, b_adv, b_ret in loader:
                b_states, b_actions, b_old_logps, b_adv, b_ret = \
                    b_states.to(device), b_actions.to(device), b_old_logps.to(device), \
                    b_adv.to(device), b_ret.to(device)
                
                action_logits, value_pred = self.agent(b_states)
                value_pred = value_pred.squeeze(-1)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = distributions.Categorical(action_probs)
                entropy = dist.entropy()
                new_logps = dist.log_prob(b_actions)
                
                surrogate_loss = self._calculate_surrogate_loss(b_old_logps, new_logps, b_adv)
                policy_loss, value_loss, entropy_loss = self._calculate_losses(
                    surrogate_loss, entropy, b_ret, value_pred
                )
                
                loss = policy_loss + 0.5 * value_loss + entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        return total_policy_loss / self.ppo_steps, total_value_loss / self.ppo_steps

    def evaluate(self, n_eval_episodes=10):
        self.agent.eval()
        
        all_rewards = []
        
        for _ in range(n_eval_episodes):
            state = self.env_test.reset()
            done_mask = np.zeros(self.env_test.num_envs, dtype=bool)
            ep_rewards = np.zeros(self.env_test.num_envs)
            
            while not done_mask.all():
                state_tensor = torch.FloatTensor(state).permute(0, 3, 1, 2).to(device)
                with torch.no_grad():
                    action_logits, _ = self.agent(state_tensor)
                    # greedy (deterministic) policy
                    actions = torch.argmax(F.softmax(action_logits, dim=-1), dim=-1).cpu().numpy()
            
                next_state, reward, done_step, info = self.env_test.step(actions)
                done_mask |= done_step
                ep_rewards += reward
                state = next_state
            
            all_rewards.extend(ep_rewards.tolist())
            
        avg_reward = np.mean(all_rewards)
        return avg_reward

    def learn(self, save_interval=1000, save_dir="checkpoints"):
        print('Called learn().')
        last_test_reward = 0.0
        os.makedirs(save_dir, exist_ok=True)

        start_episode = 1
        latest_ckpt = os.path.join(save_dir, "ppo_model_latest.pth")

        try:
            for episode in range(start_episode, self.max_episodes + 1):
                frac = 1.0 - (episode - 1) / self.max_episodes
                lr_now = self.initial_lr * frac
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_now

                train_reward, states, actions, logps, adv, ret = self.collect_trajectories()
                pol_loss, val_loss = self.update_policy(states, actions, logps, adv, ret)

                self.policy_losses.append(pol_loss)
                self.value_losses.append(val_loss)
                self.train_rewards.append(train_reward)

                if episode % self.print_interval == 0:
                    test_reward = self.evaluate(n_eval_episodes=50)
                    self.test_rewards.append(test_reward)
                    last_test_reward = test_reward

                    print(f"Ep {episode:3d} | LR {lr_now:.6f} | "
                        f"TrainR {np.mean(self.train_rewards[-self.n_trials:]):6.2f} | "
                        f"TestR {last_test_reward:6.2f} | "
                        f"PolL {np.mean(self.policy_losses[-self.n_trials:]):7.4f} | "
                        f"ValL {np.mean(self.value_losses[-self.n_trials:]):7.4f}")

                    if last_test_reward >= self.reward_threshold:
                        print(f"Reached reward threshold in {episode} episodes!")
                        self.save_model(os.path.join(save_dir, "ppo_model_final.pth"), episode)
                        break

                if episode % save_interval == 0:
                    path = os.path.join(save_dir, f"ppo_model_ep{episode}.pth")
                    self.save_model(path, episode)

        except KeyboardInterrupt:
            print("Training interrupted. Saving latest model...")

        self.save_model(latest_ckpt, episode)
        print("Training finished.")
        return self.policy_losses, self.value_losses, self.train_rewards, self.test_rewards


    def save_model(self, path=None, episode=0):
        if path is None:
            path = f"ppo_model_ep{episode}.pth"
        torch.save({
            'episode': episode,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_rewards': self.train_rewards,
            'test_rewards': self.test_rewards,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path='ppo_model.pth'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        checkpoint = torch.load(path, map_location=device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_rewards = checkpoint.get('train_rewards', [])
        self.test_rewards = checkpoint.get('test_rewards', [])
        self.policy_losses = checkpoint.get('policy_losses', [])
        self.value_losses = checkpoint.get('value_losses', [])
        self.agent.eval()
        print(f"Model loaded from {path}")
        return checkpoint


