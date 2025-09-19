from ppo import PPO
from utils import *

if __name__ == '__main__':
    ppo_agent = PPO(
        env_name="PhoenixNoFrameskip-v4",
        hidden_dimensions=512,
        discount_factor=0.99,
        gae_lambda=0.95,
        ppo_steps=10,
        epsilon=0.2,                  # smaller clip range to stabilize policy updates
        entropy_coefficient=0.01,      # increased entropy bonus for exploration
        learning_rate=2.5e-4,            # reduced learning rate to avoid premature convergence
        rollout_length=128,           # longer rollout length for better advantage estimation
        max_episodes=48828,             # number of training episodes
        reward_threshold=50000,
        print_interval=1,
        n_trials=100,
        n_envs_train=8,                # number of parallel environments for stable training
        n_envs_test=1
    )

    policy_losses, value_losses, train_rewards, test_rewards = ppo_agent.learn(
        save_interval=1000,
    )

    ppo_agent.save_model()

    plot_policy_losses(policy_losses)
    plot_value_losses(value_losses)
    plot_train_rewards(train_rewards)
    plot_test_rewards(test_rewards)
