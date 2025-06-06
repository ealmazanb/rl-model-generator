import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD
import joblib
from sklearn.preprocessing import StandardScaler


class FeatureEngineeringModule:
    def __init__(self, df):
        self.df = df.copy()

    def normalize_features(self, save_path=None):

        self.feature_cols = [col for col in self.df.columns if col != 'timestamp']
        self.scaler = StandardScaler()
        self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])
        if save_path:
            joblib.dump(self.scaler, save_path)

    def apply_technical_indicators(self, asset):
        price_col = f"{asset}_value"
        temp_df = pd.DataFrame(index=self.df.index)
        temp_df[f"{asset}_feature_ma_5"] = self.df[price_col].rolling(window=5).mean()
        temp_df[f"{asset}_feature_volatility_5"] = self.df[price_col].rolling(window=5).std()
        rsi = RSIIndicator(close=self.df[price_col], window=14)
        temp_df[f"{asset}_feature_rsi"] = rsi.rsi()
        macd = MACD(close=self.df[price_col])
        temp_df[f"{asset}_feature_macd"] = macd.macd_diff()
        self.df = pd.concat([self.df, temp_df], axis=1)

    def apply_to_all_assets(self, asset_list):
        for asset in asset_list:
            self.apply_technical_indicators(asset)

    def get_featured_data(self):
        return self.df.dropna().reset_index(drop=True)

    def scale_standard(self, save_path=None, load_path=None, exclude_value=True):

        df = self.df.copy()
        exclude = ['timestamp']
        if exclude_value:
            exclude += [col for col in df.columns if col.endswith('_value')]

        scale_cols = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]

        if load_path:
            self.scaler = joblib.load(load_path)
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(df[scale_cols])
            if save_path:
                joblib.dump(self.scaler, save_path)

        df[scale_cols] = self.scaler.transform(df[scale_cols])
        self.df = df
        return self.df

    def remove_na_rows(self):
        self.df = self.df.dropna().reset_index(drop=True)
        return self.df
