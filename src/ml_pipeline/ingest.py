import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path):
    """Load a CSV file into a DataFrame.

    Args:
        data_path: Path to the CSV file.

    Returns:
        pd.DataFrame: Raw dataset.
    """
    return pd.read_csv(data_path)


def prepare_data(df, target_column, exclude_columns):
    """Split a DataFrame into features (X) and target (y).

    Args:
        df: Raw DataFrame.
        target_column: Name of the column to predict.
        exclude_columns: Additional columns to drop (e.g. ID fields).

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.
    """
    cols_to_drop = exclude_columns + [target_column]
    X = df.drop(columns=cols_to_drop)
    y = df[target_column]
    return X, y


def split_data(X, y, test_size, random_state):
    """Split features and target into train and test sets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data to use for the test set (e.g. 0.2).
        random_state: Seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
