from ppo import PPO
from utils import *

if __name__ == '__main__':
    ppo_agent = PPO(
        env_name="PhoenixNoFrameskip-v4",
        hidden_dimensions=512,
        discount_factor=0.99,
        gae_lambda=0.95,
        ppo_steps=10,
        epsilon=0.2,
        entropy_coefficient=0.01,
        learning_rate=2.5e-4,
        rollout_length=128,
        max_episodes=48828,
        reward_threshold=50000,
        print_interval=1,
        n_trials=100,
        n_envs_train=8,
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
