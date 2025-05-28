import gymnasium as gym
from gymnasium import spaces
import numpy as np
from register.register_module import RegisterModule


class TradingEnv(gym.Env):

    def __init__(self, df, assets, initial_liquidity, reserve, tp_min, tp_max, sl_min, sl_max, epsilon=0.2,
                 use_logs=True, log_to_file=True, log_to_file_name="", min_trade_amount=200, liquidity_bonus=0.1):
        super().__init__()
        self.register = RegisterModule(f"logs/{log_to_file_name}") if log_to_file else None
        if self.register:
            self.register.set_df_columns()
        self.initial_value = None
        self.holdings = None
        self.df = df
        self.current_step = 0
        self.transaction_cost = 0.001
        self.assets = assets
        self.n_assets = len(self.assets)
        self.initial_liquidity = initial_liquidity
        self.liquidity = initial_liquidity
        self.reserve_ratio = reserve
        self.tp_min = tp_min
        self.tp_max = tp_max
        self.sl_min = sl_min
        self.sl_max = sl_max
        self.epsilon = epsilon
        self.use_logs = use_logs
        self.min_trade_amount = min_trade_amount
        self.liquidity_bonus = liquidity_bonus
        self.feature_columns = [col for col in df.columns if col != "timestamp"]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1 + self.n_assets + len(self.feature_columns),),
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.liquidity = self.initial_liquidity
        self.holdings = np.zeros(self.n_assets)
        self.initial_value = self.initial_liquidity
        self.prev_value = self.initial_value
        self.episode_returns = []
        return self._get_obs(), {}

    def _get_obs(self):
        features = self.df.iloc[self.current_step][self.feature_columns].values
        return np.concatenate(([self.liquidity], self.holdings, features), axis=0).astype(np.float32)

    def _calculate_asset_value(self):
        prices = self.df.iloc[self.current_step][[f'{asset}_value' for asset in self.assets]].values
        return np.sum(self.holdings * prices)

    def calculate_total_value(self):
        return self.liquidity + self._calculate_asset_value()

    def calculate_performance(self):
        return (self.calculate_total_value() - self.initial_value) / self.initial_value * 100

    def save_log_to_file(self):
        self.register.write_csv_to_file()

    def step(self, action):
        if self.current_step >= len(self.df):
            raise IndexError("Se llamó a step() fuera del rango del DataFrame")

        prices = self.df.iloc[self.current_step][[f'{asset}_value' for asset in self.assets]].values

        for i, act in enumerate(action):
            timestamp = self.df.iloc[self.current_step]["timestamp"]

            if prices[i] <= 0 or np.isnan(prices[i]):
                continue
            if act > self.epsilon:
                raw_amount = act * (self.liquidity * (1 - self.reserve_ratio))
                max_allocation = 0.3
                max_amount = max_allocation * self.calculate_total_value()
                amount = min(raw_amount, max_amount)
                if prices[i] > 0:
                    units = amount / prices[i]
                else:
                    units = 0.0
                if amount >= self.min_trade_amount and self.liquidity >= amount:
                    self.liquidity -= amount
                    self.holdings[i] += units
                    if self.register:
                        self.register.record_operation(
                            timestamp=timestamp,
                            code=self.assets[i],
                            price=prices[i],
                            quantity=units,
                            operation="buy",
                            liquidity=self.liquidity,
                            asset_value=self._calculate_asset_value()
                        )

            elif act < -self.epsilon:
                units = -act * self.holdings[i]
                if prices[i] > 0:
                    revenue = units * prices[i]
                else:
                    revenue = 0.0
                    units = 0.0
                performance = (prices[i] - self.df.iloc[0][f'{self.assets[i]}_value']) / self.df.iloc[0][
                    f'{self.assets[i]}_value'] * 100
                if self._can_sell(performance) and revenue >= self.min_trade_amount:
                    self.liquidity += revenue
                    self.holdings[i] -= units
                    if self.register:
                        self.register.record_operation(
                            timestamp=timestamp,
                            code=self.assets[i],
                            price=prices[i],
                            quantity=units,
                            operation="sell",
                            liquidity=self.liquidity,
                            asset_value=self._calculate_asset_value()
                        )

            else:
                pass  # hold
        self.current_step += 1

        terminated = bool(self.calculate_total_value() <= 0)
        truncated = bool(self.current_step >= len(self.df) - 1)
        new_value = self.calculate_total_value()
        returns = (new_value - self.prev_value) / self.prev_value
        reward = returns - self.transaction_cost * np.sum(np.abs(action))
        reward += self.liquidity_bonus * (self.liquidity / self.initial_liquidity)
        self.episode_returns.append(reward)
        self.prev_value = new_value
        if terminated or truncated:
            total_return = sum(self.episode_returns)
            if self.use_logs:
                print(f"[EPISODIO TERMINADO] Valor final: {self.calculate_total_value():,.2f} | Retorno acumulado: {total_return:.4f}")
        elif self.current_step % 1000 == 0:
            if self.use_logs:
                print( f"[STEP {self.current_step}] Valor total: {self.calculate_total_value():,.2f} | Liquidez: {self.liquidity:,.2f}")

        return self._get_obs(), reward, terminated, truncated, {}

    def _can_sell(self, perf):
        if self.sl_max and perf <= -self.sl_max:
            return True

        if self.tp_max and perf >= self.tp_max:
            return True

        if self.sl_min and perf > -self.sl_min:
            return False

        if self.tp_min and perf < self.tp_min:
            return False

        return True
