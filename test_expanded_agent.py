import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import our fix for pandas_ta
import fix_pandas_ta

from env.trading_env import create_trading_env
from models.dqn_agent import DQNAgent
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

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

def test_agent(model_path, stock_file, episodes=1):
    """
    Test a trained agent on a stock file and visualize the results
    
    Args:
        model_path: Path to the saved model
        stock_file: Path to the stock CSV file
        episodes: Number of episodes to run
    """
    # Track actions and performance
    actions_history = []
    portfolio_history = []
    price_history = []
    
    # Create the environment
    env = create_trading_env(stock_file, use_indicators=True)
    
    # Get stock name
    stock_name = os.path.basename(stock_file).replace('.csv', '')
    
    # Create an agent with the appropriate dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    # Load the saved model
    if not agent.load_model(model_path):
        print(f"Failed to load model from {model_path}")
        return
    
    print(f"Testing agent on {stock_name}...")
    
    # Set epsilon to a small value for some exploration
    agent.epsilon = 0.05
    
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        initial_value = info['portfolio_valuation']
        
        # Reset history for each episode
        actions_history = []
        portfolio_history = [initial_value]
        price_history = [info['last_price']]
        
        while not done and not truncated:
            # Select action
            action = agent.select_action(state)
            
            # Record the action
            actions_history.append(action)
            
            # Step the environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Record portfolio value and price
            portfolio_history.append(info['portfolio_valuation'])
            price_history.append(info['last_price'])
            
            # Update state and reward
            state = next_state
            total_reward += reward
            step += 1
        
        # Calculate return
        final_value = info['portfolio_valuation']
        episode_return = ((final_value - initial_value) / initial_value) * 100
        
        print(f"Episode {episode+1} Results:")
        print(f"  Initial Portfolio: ${initial_value:.2f}")
        print(f"  Final Portfolio: ${final_value:.2f}")
        print(f"  Return: {episode_return:.2f}%")
        print(f"  Total Steps: {step}")
        
        # Count actions
        action_counts = {}
        for action in actions_history:
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
        
        print("\nActions Taken:")
        for action, count in action_counts.items():
            action_name = ACTION_MEANINGS.get(action, f"Action {action}")
            percentage = (count / len(actions_history)) * 100
            print(f"  {action_name}: {count} ({percentage:.1f}%)")
        
        # Visualize the results
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Portfolio Value
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_history, label='Portfolio Value')
        plt.title(f'Portfolio Value Over Time for {stock_name}')
        plt.xlabel('Step')
        plt.ylabel('Value ($)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot 2: Stock Price and Actions
        plt.subplot(2, 1, 2)
        plt.plot(price_history, label='Stock Price', color='blue')
        
        # Add markers for different actions
        for i, action in enumerate(actions_history):
            action_name = ACTION_MEANINGS.get(action, f"Action {action}")
            
            # Choose color and marker based on action
            if action == 0:  # Short
                color, marker = 'red', 'v'
            elif action == 2:  # Long
                color, marker = 'green', '^'
            elif action == 3:  # Buy more
                color, marker = 'darkgreen', '>'
            elif action == 4:  # Sell all
                color, marker = 'orange', 'x'
            elif action == 6:  # Buy 2x size
                color, marker = 'purple', '*'
            else:  # Hold or Do nothing
                continue  # Skip markers for hold/do nothing to reduce clutter
                
            plt.plot(i+1, price_history[i+1], marker=marker, color=color, 
                     markersize=8, label=action_name if i == 0 else "")
        
        # Add a custom legend for actions
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='v', color='w', markerfacecolor='red', label='Short', markersize=8),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='green', label='Long', markersize=8),
            Line2D([0], [0], marker='>', color='w', markerfacecolor='darkgreen', label='Buy more', markersize=8),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='orange', label='Sell all', markersize=8),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='purple', label='Buy 2x', markersize=8),
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.title(f'Stock Price and Trading Actions for {stock_name}')
        plt.xlabel('Step')
        plt.ylabel('Price ($)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Check if saved models directory exists
    saved_models_dir = "saved_models"
    if not os.path.exists(saved_models_dir):
        print(f"No saved models found in {saved_models_dir}.")
        exit(1)
    
    # Find the most recent model
    model_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.pt')]
    if not model_files:
        print("No model files found.")
        exit(1)
    
    # Sort by episode number (assuming format dqn_agent_ep{num}.pt)
    model_files.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]), reverse=True)
    latest_model = os.path.join(saved_models_dir, model_files[0])
    
    print(f"Testing with the latest model: {latest_model}")
    
    # Test on all stock files
    data_dir = "data"
    stock_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Randomly select one stock file
    stock_file = np.random.choice(stock_files)
    
    test_agent(latest_model, stock_file, episodes=1) 