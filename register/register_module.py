import pandas as pd

rl_operation_columns = ['id', 'timestamp', 'code', 'price', 'quantity', 'operation', 'liquidity', 'asset_value']


class RegisterModule:

    def __init__(self, file_name):
        self.file_name = file_name
        self.operations = pd.DataFrame(columns=rl_operation_columns)
        self.counter = 0

    def record_operation(self, timestamp, code, price, quantity, operation, liquidity, asset_value):
        row = {
            'id': self.counter,
            'timestamp': timestamp,
            'code': code.strip(),
            'price': float(price),
            'quantity': float(quantity),
            'operation': operation.lower().strip(),
            'liquidity': float(liquidity),
            'asset_value': float(asset_value)
        }
        self.operations.loc[len(self.operations)] = row
        self.counter += 1

    def set_df_columns(self, columns=None):
        if columns is None:
            columns = rl_operation_columns
        self.operations = pd.DataFrame(columns=columns)

    def write_csv_to_file(self):
        self.operations.to_csv(self.file_name, index=False)