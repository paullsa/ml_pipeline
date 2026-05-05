# data loading and validation
# ingest.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def prepare_data(df, target_column, exclude_columns):
    cols_to_drop = exclude_columns + [target_column]
    X = df.drop(columns=cols_to_drop)
    y = df[target_column]
    return X, y

def split_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test