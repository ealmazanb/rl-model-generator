from stable_baselines3 import PPO
from environment import TradingEnv
from utils import Utils

"""
Data definition
"""
BASE_INPUT_DIR = 'input'
SENTIMENT_DIR = 'sentiment'
MACRO_DIR = 'macro'
HISTORIC_DIR = 'historic'
MODEL_OUTPUT = 'models'
FILES_DIR = 'files'

initial_liquidity = 100_000
reserve = 0.1
TP_MIN = 0
TP_MAX = 9999999
SL_MIN = 0
SL_MAX = 9999999

test_date_range_start = '2000-01-01'
test_date_range_end = '2025-01-01'

policy = 'MlpPolicy'

log_filename = 'ppo_trading_agent_v1_00_25'
model_name = 'ppo_trading_agent_v2_15M'

utils = Utils()
df = utils.load_dataset(f'{FILES_DIR}/full_dataset.csv')
df_test = utils.df_date_filter(test_date_range_start, test_date_range_end, df)

assets = ['gold', 'apple', 'boeing', 'johnson_johnson', 'oil', 'intel']

model = PPO.load(f'{MODEL_OUTPUT}/{model_name}')

env_test = TradingEnv(
    df=df_test,
    assets=assets,
    initial_liquidity=initial_liquidity,
    reserve=reserve,
    tp_min=TP_MIN,
    tp_max=TP_MAX,
    sl_min=SL_MIN,
    sl_max=SL_MAX,
    log_to_file=True,
    log_to_file_name=log_filename
)

obs, _ = env_test.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, _ = env_test.step(action)

final_value = env_test.calculate_total_value()
performance = env_test.calculate_performance()

print(f"Last testing value: {final_value:.2f}")
print(f"Test performance: {env_test.calculate_performance():.2f}%")

env_test.save_log_to_file()
