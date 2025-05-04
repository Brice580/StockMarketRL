import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np

def plot_all_stocks(stock_files):
    """
    Plot all stocks in the data directory together for comparison.
    
    Args:
        stock_files: List of paths to stock CSV files
    """
    plt.figure(figsize=(14, 8))
    
    for stock_path in stock_files:
        try:
            stock_name = os.path.basename(stock_path).replace('.csv', '')
            
            # Read the stock data
            df = pd.read_csv(stock_path)
            
            # Convert the first column to datetime if it's a date
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                x = df['Date']
            else:
                x = range(len(df))
            
            # Normalize the closing price to start at 100 for better comparison
            if 'Close' in df.columns:
                # Convert to numeric, coerce errors to NaN
                close_prices = pd.to_numeric(df['Close'], errors='coerce')
            else:
                # Assume second column is close price
                close_prices = pd.to_numeric(df.iloc[:, 1], errors='coerce')
            
            # Drop any NaN values that couldn't be converted
            close_prices = close_prices.dropna()
            
            if len(close_prices) == 0:
                print(f"Warning: No valid numeric prices found for {stock_name}, skipping...")
                continue
                
            if close_prices.iloc[0] == 0:
                print(f"Warning: First price value is 0 for {stock_name}, skipping...")
                continue
                
            normalized_prices = (close_prices / close_prices.iloc[0]) * 100
            
            # Plot the normalized stock prices (without label for legend)
            line, = plt.plot(x[:len(normalized_prices)], normalized_prices, linewidth=1.5)
            
            # Add the label directly at the end of the line
            last_x = x[:len(normalized_prices)].iloc[-1] if hasattr(x, 'iloc') else x[len(normalized_prices)-1]
            last_y = normalized_prices.iloc[-1]
            
            # Calculate the final return percentage
            return_pct = ((last_y / 100) - 1) * 100
            label = f"{stock_name} ({return_pct:.0f}%)"
            
            # Add annotation with stock name and return percentage
            plt.annotate(label, 
                         xy=(last_x, last_y),
                         xytext=(5, 0), 
                         textcoords='offset points',
                         va='center',
                         color=line.get_color(),
                         fontweight='bold')
            
        except Exception as e:
            print(f"Error processing {stock_name}: {e}")
    
    plt.title('Normalized Stock Price Comparison (Log Scale, Starting at 100)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Price (log scale)', fontsize=12)
    # No legend needed since labels are placed directly on the chart
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Use logarithmic scale for y-axis
    plt.yscale('log')
    
    # Format the x-axis to show dates nicely if we have dates
    if 'Date' in df.columns:
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Find all CSV files in the data directory
    data_dir = "data"
    stock_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if stock_files:
        print(f"Found {len(stock_files)} stock files:")
        for file in stock_files:
            print(f"  - {os.path.basename(file)}")
        plot_all_stocks(stock_files)
    else:
        print("No CSV files found in the data directory.")