from env.trading_env import create_trading_env
from models.dqn_agent import DQNAgent
from models.dqn_agent_finetune import DQNAgentFinetuned
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import defaultdict

USE_TECHNICAL_INDICATORS = True 

def main():
    stock_files = [
        "data/AAPL.csv", "data/MSFT.csv", "data/GOOGL.csv", "data/NVDA.csv",
        "data/AMZN.csv", "data/UNH.csv", "data/XOM.csv", "data/V.csv",
        "data/PG.csv", "data/META.csv", "data/MA.csv", "data/LLY.csv",
        "data/JPM.csv", "data/JNJ.csv", "data/BRK-B.csv",
    ]

    num_episodes = 5
    
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

    plt.style.use('bmh')  
    fig = plt.figure(figsize=(20, 12))

    # 1. Portfolio Value Over Time
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(performance_metrics['portfolio_values'], label='Portfolio Value', color='blue')
    ax1.set_title('Portfolio Value Progression')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True)

    # 2. Rewards Distribution by Stock
    ax2 = plt.subplot(2, 3, 2)
    stock_means = {k: np.mean(v) for k, v in performance_metrics['stock_performance'].items()}
    stock_stds = {k: np.std(v) for k, v in performance_metrics['stock_performance'].items()}
    stocks = list(stock_means.keys())
    means = list(stock_means.values())
    stds = list(stock_stds.values())
    x_pos = np.arange(len(stocks))
    ax2.bar(x_pos, means, yerr=stds, capsize=5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stocks, rotation=45, ha='right')
    ax2.set_title('Average Reward by Stock')
    ax2.set_ylabel('Mean Reward')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # 3. Action Distribution
    ax3 = plt.subplot(2, 3, 3)
    actions = list(performance_metrics['actions_taken'].keys())
    counts = list(performance_metrics['actions_taken'].values())
    ax3.pie(counts, labels=['Short', 'Hold', 'Long'], autopct='%1.1f%%')
    ax3.set_title('Action Distribution')

    # 4. Returns Distribution
    ax4 = plt.subplot(2, 3, 4)
    all_returns = [ret for returns in performance_metrics['returns_by_stock'].values() for ret in returns]
    ax4.hist(all_returns, bins=30, edgecolor='black')
    ax4.axvline(np.mean(all_returns), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(all_returns):.2f}%')
    ax4.set_title('Returns Distribution')
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    # 5. Episode Length Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(performance_metrics['episode_lengths'], bins=20, edgecolor='black')
    ax5.set_title('Episode Length Distribution')
    ax5.set_xlabel('Number of Steps')
    ax5.set_ylabel('Frequency')

    # 6. Learning Progress (Moving Average of Returns)
    ax6 = plt.subplot(2, 3, 6)
    window_size = 5
    returns = [ret for returns in performance_metrics['returns_by_stock'].values() for ret in returns]
    moving_avg = pd.Series(returns).rolling(window=window_size).mean()
    ax6.plot(moving_avg, label=f'{window_size}-Episode Moving Average')
    ax6.set_title('Learning Progress')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Average Return (%)')
    ax6.legend()

    plt.tight_layout()
    plt.show()

    # Additional Statistics
    print("\nTraining Summary:")
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Return: {np.mean(all_returns):.2f}%")
    print(f"Best Return: {np.max(all_returns):.2f}%")
    print(f"Worst Return: {np.min(all_returns):.2f}%")
    print(f"Average Episode Length: {np.mean(performance_metrics['episode_lengths']):.1f} steps")
    
    # Best and Worst Performing Stocks
    stock_avg_returns = {k: np.mean(v) for k, v in performance_metrics['returns_by_stock'].items()}
    best_stock = max(stock_avg_returns.items(), key=lambda x: x[1])
    worst_stock = min(stock_avg_returns.items(), key=lambda x: x[1])
    print(f"\nBest Performing Stock: {best_stock[0]} ({best_stock[1]:.2f}%)")
    print(f"Worst Performing Stock: {worst_stock[0]} ({worst_stock[1]:.2f}%)")

if __name__ == "__main__":
    main()
