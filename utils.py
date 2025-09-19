import matplotlib.pyplot as plt
import os

def plot_policy_losses(policy_losses, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(policy_losses)
    plt.title("Policy Losses")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_dir, "policy_losses.png"))
    plt.close()


def plot_value_losses(value_losses, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(value_losses)
    plt.title("Value Losses")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_dir, "value_losses.png"))
    plt.close()


def plot_train_rewards(train_rewards, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_rewards)
    plt.title("Train Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(save_dir, "train_rewards.png"))
    plt.close()


def plot_test_rewards(test_rewards, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(test_rewards)
    plt.title("Test Rewards")
    plt.xlabel("Evaluation step")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(save_dir, "test_rewards.png"))
    plt.close()
