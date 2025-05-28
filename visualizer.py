import pandas as pd
import matplotlib.pyplot as plt

log_filename = 'ppo_trading_agent_v1_00_25'
df = pd.read_csv(f'logs/{log_filename}', parse_dates=['timestamp'])
df['total_value'] = df['liquidity'] + df['asset_value']
df['year'] = df['timestamp'].dt.year
annual = df.groupby('year').last()[['liquidity', 'asset_value', 'total_value']]
plt.figure(figsize=(12, 6))
annual[['liquidity', 'asset_value', 'total_value']].plot(kind='bar')
plt.title('Year evolution')
plt.ylabel('Value')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
annual['pct_change'] = annual['total_value'].pct_change() * 100
plt.figure(figsize=(10, 5))
annual['pct_change'].plot(kind='bar', color='skyblue')
plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='No change (%)')
plt.title('Percent year change %')
plt.ylabel('%')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
df_ops = pd.read_csv(f'logs/{log_filename}', parse_dates=['timestamp'])
df_ops['year'] = df_ops['timestamp'].dt.year
operation_summary = df_ops.groupby(['year', 'operation']).size().unstack(fill_value=0)
print('Operations')
print(operation_summary)
top_assets = df_ops['code'].value_counts().head(10)
plt.figure(figsize=(8, 8))
top_assets.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Op. distributions')
plt.ylabel('')
plt.tight_layout()
plt.show()

df_ops = df_ops.sort_values(by='timestamp').reset_index(drop=True)

avg_buy_price = {}
profitable_sales = 0
unprofitable_sales = 0

for _, row in df_ops.iterrows():
    code, op, price, qty = row['code'], row['operation'], row['price'], row['quantity']

    if op == 'buy':
        if code not in avg_buy_price:
            avg_buy_price[code] = {'total_qty': 0.0, 'total_spent': 0.0}
        avg_buy_price[code]['total_qty'] += qty
        avg_buy_price[code]['total_spent'] += price * qty

    elif op == 'sell' and code in avg_buy_price and avg_buy_price[code]['total_qty'] > 0:
        avg_price = avg_buy_price[code]['total_spent'] / avg_buy_price[code]['total_qty']
        if price > avg_price:
            profitable_sales += 1
        else:
            unprofitable_sales += 1
        avg_buy_price[code]['total_qty'] -= qty
        avg_buy_price[code]['total_spent'] -= avg_price * qty

avg_buy_price = {}
profits = []

for _, row in df_ops.iterrows():
    code, op, price, qty = row['code'], row['operation'], row['price'], row['quantity']

    if op == 'buy':
        if code not in avg_buy_price:
            avg_buy_price[code] = {'total_qty': 0.0, 'total_spent': 0.0}
        avg_buy_price[code]['total_qty'] += qty
        avg_buy_price[code]['total_spent'] += price * qty

    elif op == 'sell' and code in avg_buy_price and avg_buy_price[code]['total_qty'] > 0:
        avg_price = avg_buy_price[code]['total_spent'] / avg_buy_price[code]['total_qty']
        profit = (price - avg_price) * qty
        profits.append(profit)
        avg_buy_price[code]['total_qty'] -= qty
        avg_buy_price[code]['total_spent'] -= avg_price * qty

print(f"Best gain: {max(profits):,.2f}")
print(f"Worst loss: {min(profits):,.2f}")

