import pandas as pd
import numpy as np
from qplex import QModel
from qplex.model.execution_config import ExecutionConfig


def get_data():
    # Load your portfolio data
    data = pd.read_csv('data/portfolio_data.csv', index_col=0)
    # Calculate returns and volatility
    monthly_returns = data[data.columns.values.tolist()].pct_change().iloc[1:]
    avg_monthly_returns = monthly_returns.mean(axis=0)
    volatility = monthly_returns.std(axis=0)

    return avg_monthly_returns, volatility, data.columns.tolist()


# Get data
avg_monthly_returns, volatility, stocks = get_data()

# Limit to small number of stocks and convert to integers by scaling
n = 8  # Use only 8 stocks for simplicity
stocks = stocks[:n]

# Scale returns and volatility to integers (multiply by 10000 and round)
scaling_factor = 10000
returns_scaled = np.round(avg_monthly_returns[stocks] * scaling_factor).astype(
    int)
volatility_scaled = np.round(volatility[stocks] * scaling_factor).astype(int)

# Create model
print("Creating QModel")
portfolio_model = QModel('portfolio')

# Define binary variables (1 if stock is selected, 0 otherwise)
print("Adding variables...")
x = [portfolio_model.binary_var(name=f"select_{stocks[i]}") for i in range(n)]

# Cardinality constraint - select exactly 4 stocks
k = 4  # Number of stocks to select
print("Adding cardinality constraint...")
portfolio_model.add_constraint(portfolio_model.sum(x) == k)

# Define objective: maximize returns while penalizing risk
# Use a parameter to balance return vs. risk
risk_aversion = 1.5  # Adjust this parameter to change the risk/return tradeoff

# Linear objective function with integer coefficients
print("Setting objective...")
objective = sum(
    returns_scaled[i] * x[i] for i in range(n)) - risk_aversion * sum(
    volatility_scaled[i] * x[i] for i in range(n))
portfolio_model.set_objective('max', objective)

print("Solving...")
execution_config = ExecutionConfig(
    provider="ibmq",
    backend="simulator",
    algorithm="qao-ansatz",  # Using standard QAOA instead of QAO-ansatz
    p=2,  # Reduced p-value for faster execution
    shots=1024,
    max_iter=100,  # Reduced iterations
)
portfolio_model.solve('quantum', execution_config)
print("Done")

# Print solution and interpret results
print("\nSelected stocks:")
total_return = 0
total_risk = 0
for i in range(n):
    if portfolio_model.solution.get_value(
            x[i]) > 0.5:  # Check if binary variable is set to 1
        stock_return = avg_monthly_returns[stocks[i]]
        stock_vol = volatility[stocks[i]]
        total_return += stock_return
        total_risk += stock_vol
        print(
            f"{stocks[i]}: Expected Return = {stock_return:.6f}, Volatility "
            f"= {stock_vol:.6f}")

print(f"\nPortfolio Statistics:")
print(f"Total Expected Monthly Return: {total_return:.6f}")
print(f"Total Risk (sum of volatilities): {total_risk:.6f}")
print(
    f"Return/Risk Ratio: {total_return / total_risk if total_risk else 0:.6f}")