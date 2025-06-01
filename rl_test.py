import os

from stable_baselines3 import PPO
from environment import TradingEnv
from feature import FeatureEngineeringModule
from ingestion import DataIngestionModule
from utils import Utils


"""
Data definition
"""
BASE_INPUT_DIR = 'input'
SENTIMENT_DIR = 'sentiment'
MACRO_DIR = 'macro'
HISTORIC_DIR = 'historic'
MODEL_OUTPUT = 'models'
SAVES_CALLBACK_DIR = 'models/callback'
FILES_DIR = 'files'

initial_liquidity = 100_000
reserve = 0.1
TP_MIN = 0
TP_MAX = 9999999
SL_MIN = 0
SL_MAX = 9999999

test_date_start = '2001-01-02'
test_date_end = '2025-01-01'

policy = 'MlpPolicy'

log_filename = 'ppo_trading_agent_v4_00_25'
model_name = 'ppo_trading_agent_v4_15M_scaled'

'''
Data ingestion
'''
sentiment_files = {
    'alibaba': 'alibaba.csv',
    'apple': 'apple.csv',
    'astrazeneca': 'astrazeneca.csv',
    'bitcoin': 'bitcoin.csv',
    'boeing': 'boeing.csv',
    'coca_cola': 'coca_cola.csv',
    'ethereum': 'ethereum.csv',
    'eu_bonds': 'eu_bonds.csv',
    'gold': 'gold.csv',
    'intel': 'intel.csv',
    'johnson_johnson': 'johnson_johnson.csv',
    'lockheed_martin': 'lockheed_martin.csv',
    'oil': 'oil.csv',
    'petrobras': 'petrobras.csv',
    'tesla': 'tesla.csv',
    'us_bonds': 'us_bonds.csv',
}

macro_files = {
    'usa_gdp': 'usa_gdp.csv',
    'usa_inflation': 'usa_inflation.csv',
    'usa_unemployment': 'usa_unemployment.csv',
    'usa_interest_rate': 'usa_interest_rate.csv',
    'euro_gdp': 'euro_gdp.csv',
    'euro_inflation': 'euro_inflation.csv',
    'euro_unemployment': 'euro_unemployment.csv',
    'euro_interest_rate': 'euro_interest_rate.csv',
}

historical_asset_files = {
    'alibaba': 'alibaba.csv',
    'apple': 'apple.csv',
    'astrazeneca': 'astrazeneca.csv',
    'bitcoin': 'bitcoin.csv',
    'boeing': 'boeing.csv',
    'coca_cola': 'coca_cola.csv',
    'ethereum': 'ethereum.csv',
    'eu_bonds': 'eu_bonds.csv',
    'gold': 'gold.csv',
    'intel': 'intel.csv',
    'johnson_johnson': 'johnson_johnson.csv',
    'lockheed_martin': 'lockheed_martin.csv',
    'oil': 'oil.csv',
    'petrobras': 'petrobras.csv',
    'tesla': 'tesla.csv',
    'us_bonds': 'us_bonds.csv',
}
utils = Utils()

assets = ['gold', 'apple', 'petrobras', 'astrazeneca', 'coca_cola', 'lockheed_martin']
sentiment_paths = {k: os.path.join(SENTIMENT_DIR, v) for k, v in sentiment_files.items()}
macro_paths = {k: os.path.join(MACRO_DIR, v) for k, v in macro_files.items()}
asset_paths = {k: os.path.join(HISTORIC_DIR, v) for k, v in historical_asset_files.items()}
ingestor = DataIngestionModule(data_dir=BASE_INPUT_DIR)
ingestor.ingest_macro_data(macro_paths)
ingestor.ingest_asset_data(asset_paths)
ingestor.ingest_sentiment_data(sentiment_paths)
df = ingestor.align_all_data()

'''
Part II - Feature engineering
'''
feature_module = FeatureEngineeringModule(df)
feature_module.apply_to_all_assets(assets)
feature_module.remove_na_rows()
feature_module.scale_standard(save_path="files/scaler.pkl")
df = feature_module.get_featured_data()
df_test = utils.df_date_filter(
    from_date=test_date_start,
    to_date=test_date_end,
    df=df
)

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
