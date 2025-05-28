import pandas as pd
from datetime import datetime


class Utils:

    def __init__(self, debug=True):
        self.debug = debug

    def log(self, log):
        if self.debug:
            print(log)

    def df_date_filter(self, from_date, to_date, df):
        df = df.copy()
        try:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d")
            to_dt = datetime.strptime(to_date, "%Y-%m-%d")
        except ValueError as e:
            self.log(f"Error al convertir fechas: {e}")
            return df

        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d")
        except Exception as e:
            return df

        filtered_df = df[(df['timestamp'] >= from_dt) & (df['timestamp'] <= to_dt)]
        return filtered_df

    def shift_column(self, column, shift_by, new_column, df):
        df = df.copy()
        df[new_column] = df[column]
        df[new_column] = df[new_column].shift(shift_by)
        df = df.dropna(subset=[new_column])
        return df

    def shift_window(self, target_column, window, df):
        df = df.copy()
        for w in range(window, 0, -1):
            for col in df.columns:
                if col != target_column and col != "timestamp":
                    df[f"{col}t-{w}"] = df[col].shift(w)
        return df

    def load_dataset(self, path):
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        return df