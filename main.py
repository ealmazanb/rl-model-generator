import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from feature import FeatureEngineeringModule
from ingestion import DataIngestionModule
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
SAVES_CALLBACK_DIR = 'models/callback'
FILES_DIR = 'files'

initial_liquidity = 100_000
reserve = 0.1
TP_MIN = 0
TP_MAX = 9999999
SL_MIN = 0
SL_MAX = 9999999

train_date_start = '2000-01-01'
train_date_end = '2025-01-01'

test_date_start = '2022-01-02'
test_date_end = '2025-01-01'

steps = 15_000_000
policy = 'MlpPolicy'

log_filename = 'ppo_trading_agent_v4_00_22'
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

assets = ['gold', 'apple', 'boeing', 'johnson_johnson', 'oil', 'intel']
sentiment_paths = {k: os.path.join(SENTIMENT_DIR, v) for k, v in sentiment_files.items()}
macro_paths = {k: os.path.join(MACRO_DIR, v) for k, v in macro_files.items()}
asset_paths = {k: os.path.join(HISTORIC_DIR, v) for k, v in historical_asset_files.items()}
ingestor = DataIngestionModule(data_dir=BASE_INPUT_DIR)
ingestor.ingest_macro_data(macro_paths)
ingestor.ingest_asset_data(asset_paths)
df = ingestor.align_all_data()
df_test = utils.df_date_filter(
    from_date=test_date_start,
    to_date=test_date_end,
    df=df
)
'''
Part II - Feature engineering
'''
feature_module = FeatureEngineeringModule(df)
feature_module.apply_to_all_assets(assets)
feature_module.remove_na_rows()
feature_module.scale_standard(save_path="files/scaler.pkl")
df = feature_module.get_featured_data()

'''
Part III Environment definition
'''

df_train = utils.df_date_filter(
    from_date=train_date_start,
    to_date=train_date_end,
    df=df
)

env = TradingEnv(
    df=df_train,
    assets=assets,
    initial_liquidity=initial_liquidity,
    reserve=reserve,
    tp_min=TP_MIN,
    tp_max=TP_MAX,
    sl_min=SL_MIN,
    sl_max=SL_MAX,
    log_to_file=False
)

check_env(
    env=env,
    warn=True
)

'''
Part IV - Agent

Params are:
Param Policy MlpPolicy in this case. Others are CnnPolicy (Best for image) MultiInputPolicy for different separated 
inputs
Param env: The trading environment
Param device: The hardware used to train. Cuda forces to use GPU, but only if is NVidia GPU, if not it will use cpu
Param verbose: 0 Does not print to console, 1 Print only basic information, 2 Print all information
Param ent_coef: Usually 0-0.3 if high the model will be more explorative, if lower will be more conservative
Param learning_rate: Controls how big are steps during training. Lower is slower to learn but more precise, Higher is 
faster but have the risk of not finding optimal solutions (default 0.0003)
Param batch_size: Control the batches used to train the internal neural network while optimizing. The bigger, less 
updates per epoch and stable training but slower corrections (128, 256, 512)
'''
model = PPO(
    policy='MlpPolicy',
    env=env,
    device="cuda",
    verbose=1,
    n_steps=4096,
    batch_size=128,
    learning_rate=3.3e-4,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gamma=0.995,
    gae_lambda=0.95,
)

save_callback = CheckpointCallback(
    save_freq=1_000_000,
    save_path=SAVES_CALLBACK_DIR,
    name_prefix="ppo_checkpoint"
)

model.learn(total_timesteps=steps, callback=save_callback)
model.save(f'{MODEL_OUTPUT}/{model_name}')

'''
Part V - Test
'''


feature_module = FeatureEngineeringModule(df_test)
feature_module.apply_to_all_assets(assets)
feature_module.remove_na_rows()

feature_module.scale_standard(load_path="files/scaler.pkl")
df_test = feature_module.get_featured_data()

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
    log_to_file_name=log_filename)

obs, _ = env_test.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env_test.step(action)

final_value = env_test.calculate_total_value()
env_test.save_log_to_file()

print(f"Last testing value: {final_value:.2f}")
print(f"Test performance: {env_test.calculate_performance():.2f}%")
