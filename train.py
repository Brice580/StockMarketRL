import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import our fix for pandas_ta
import fix_pandas_ta

from env.trading_env import create_trading_env
from models.dqn_agent import DQNAgent
from models.dqn_agent_finetune import DQNAgentFinetuned
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from collections import defaultdict

print("hi")

USE_TECHNICAL_INDICATORS = True 

# Define action meanings
ACTION_MEANINGS = {
    0: "Short",       # -1: short position
    1: "Hold",        # 0: hold/no position
    2: "Long",        # 1: long position
    3: "Buy more",    # 2: increase position size (scale-in)
    4: "Sell all",    # 3: fully exit position
    5: "Do nothing",  # 4: explicit idle step
    6: "Buy 2x size"  # 5: aggressive entry (double size)
}

def main():
    stock_files = [
        "data/AAPL.csv",
    ]

    num_episodes = 10
    
    performance_metrics = {
        'episode_rewards': [],
        'portfolio_values': [],
        'actions_taken': defaultdict(int),  # Count of each action
        'stock_performance': defaultdict(list),  # Performance by stock
        'episode_lengths': [],
        'stock_sequence': [],  # Track which stock was traded in each episode
        'returns_by_stock': defaultdict(list),  # Returns for each stock
    }
    
    agent = None

    for episode in range(num_episodes):
        stock_path = np.random.choice(stock_files)
        stock_name = os.path.basename(stock_path).replace('.csv', '')
        performance_metrics['stock_sequence'].append(stock_name)
        
        env = create_trading_env(stock_path, use_indicators=USE_TECHNICAL_INDICATORS)

        if agent is None:
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            agent = DQNAgent(state_dim, action_dim)

        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        episode_actions = []
        initial_value = info['portfolio_valuation']
        step_count = 0

        while not done and not truncated:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            agent.buffer.push(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward
            episode_actions.append(action)
            step_count += 1

        final_value = info['portfolio_valuation']
        episode_return = ((final_value - initial_value) / initial_value) * 100
        performance_metrics['episode_rewards'].append(total_reward)
        performance_metrics['portfolio_values'].append(final_value)
        performance_metrics['episode_lengths'].append(step_count)
        performance_metrics['stock_performance'][stock_name].append(total_reward)
        performance_metrics['returns_by_stock'][stock_name].append(episode_return)
        
        for action in episode_actions:
            performance_metrics['actions_taken'][action] += 1

        agent.update_target()
        print(f"Episode {episode+1}: Stock = {stock_name}")
        print(f"  Total Reward = {total_reward:.2f}")
        print(f"  Final Portfolio Value = ${final_value:.2f}")
        print(f"  Return = {episode_return:.2f}%")
        print(f"  Steps = {step_count}")
        print("-" * 50)

    # Visualization Section
    plt.style.use('default')  # Reset to default style
    fig = plt.figure(figsize=(20, 12))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6', '#1abc9c']

    # 1. Portfolio Value Over Time
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(performance_metrics['portfolio_values'], color=colors[0], linewidth=2)
    ax1.set_title('Portfolio Value Progression', fontsize=12, pad=15)
    ax1.set_xlabel('Episode', fontsize=10)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. Rewards Distribution by Stock
    ax2 = plt.subplot(2, 3, 2)
    stock_means = {k: np.mean(v) for k, v in performance_metrics['stock_performance'].items()}
    stock_stds = {k: np.std(v) for k, v in performance_metrics['stock_performance'].items()}
    stocks = list(stock_means.keys())
    means = list(stock_means.values())
    stds = list(stock_stds.values())
    x_pos = np.arange(len(stocks))
    bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, color=colors[1], alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stocks, rotation=45, ha='right')
    ax2.set_title('Average Reward by Stock', fontsize=12, pad=15)
    ax2.set_ylabel('Mean Reward', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 3. Action Distribution
    ax3 = plt.subplot(2, 3, 3)
    actions = list(performance_metrics['actions_taken'].keys())
    counts = list(performance_metrics['actions_taken'].values())
    
    # Create labels using the action meanings
    action_labels = [ACTION_MEANINGS.get(action, f"Action {action}") for action in actions]
    
    wedges, texts, autotexts = ax3.pie(counts, labels=action_labels, 
                                      autopct='%1.1f%%', colors=colors[3:3+len(actions)],
                                      textprops={'fontsize': 10})
    ax3.set_title('Action Distribution', fontsize=12, pad=15)
    plt.setp(autotexts, size=9, weight="bold")

    # 4. Returns Distribution
    ax4 = plt.subplot(2, 3, 4)
    all_returns = [ret for returns in performance_metrics['returns_by_stock'].values() for ret in returns]
    if len(all_returns) > 0:  # Only plot if we have data
        ax4.hist(all_returns, bins=min(15, len(all_returns)), color=colors[2], alpha=0.7, edgecolor='white')
        mean_return = np.mean(all_returns)
        ax4.axvline(mean_return, color=colors[0], linestyle='dashed', linewidth=2, 
                    label=f'Mean: {mean_return:.2f}%')
        ax4.set_title('Returns Distribution', fontsize=12, pad=15)
        ax4.set_xlabel('Return (%)', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.legend(fontsize=9)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.grid(True, linestyle='--', alpha=0.3)

    # 5. Episode Length Distribution
    ax5 = plt.subplot(2, 3, 5)
    if len(performance_metrics['episode_lengths']) > 0:  # Only plot if we have data
        ax5.hist(performance_metrics['episode_lengths'], 
                bins=min(10, len(performance_metrics['episode_lengths'])), 
                color=colors[4], alpha=0.7, edgecolor='white')
        ax5.set_title('Episode Length Distribution', fontsize=12, pad=15)
        ax5.set_xlabel('Number of Steps', fontsize=10)
        ax5.set_ylabel('Frequency', fontsize=10)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.grid(True, linestyle='--', alpha=0.3)

    # 6. Learning Progress
    ax6 = plt.subplot(2, 3, 6)
    returns = [ret for returns in performance_metrics['returns_by_stock'].values() for ret in returns]
    if len(returns) > 1:  # Need at least 2 points for a line
        window_size = min(2, len(returns))  # Adjust window size based on data length
        moving_avg = pd.Series(returns).rolling(window=window_size, min_periods=1).mean()
        ax6.plot(moving_avg, color=colors[5], linewidth=2, 
                 label=f'{window_size}-Episode Moving Average')
        ax6.set_title('Learning Progress', fontsize=12, pad=15)
        ax6.set_xlabel('Episode', fontsize=10)
        ax6.set_ylabel('Average Return (%)', fontsize=10)
        ax6.legend(fontsize=9)
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.show()

    # Print statistics with better formatting
    print("\n" + "="*50)
    print("Training Summary".center(50))
    print("="*50)
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Return: {np.mean(all_returns):.2f}%")
    print(f"Best Return: {np.max(all_returns):.2f}%")
    print(f"Worst Return: {np.min(all_returns):.2f}%")
    print(f"Average Episode Length: {np.mean(performance_metrics['episode_lengths']):.1f} steps")
    
    print("\nStock Performance:")
    print("-"*50)
    stock_avg_returns = {k: np.mean(v) for k, v in performance_metrics['returns_by_stock'].items()}
    sorted_stocks = sorted(stock_avg_returns.items(), key=lambda x: x[1], reverse=True)
    for stock, avg_return in sorted_stocks:
        print(f"{stock:8} : {avg_return:+.2f}%")
    
    # Save the trained model
    if agent is not None:
        os.makedirs('saved_models', exist_ok=True)
        model_path = os.path.join('saved_models', f'dqn_agent_ep{num_episodes}.pt')
        agent.save_model(model_path)
        print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
