import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import pickle

categorical_columns = ['experience_level', 'employment_type', 'job_title', 'salary_currency', 'employee_residence',
                       'company_location', 'company_size']

continuous_columns = ['work_year', 'remote_ratio', 'salary_in_usd']


def prepare_dataframe_to_train(train_df_path: str):
    df = pd.read_csv(train_df_path)

    # drop the 'salary' column
    df.drop(columns=['salary'], inplace=True)

    # Get all categorical features
    os.makedirs('encoders', exist_ok=True)
    for categorical_feature in categorical_columns:
        le = LabelEncoder()
        df[categorical_feature] = le.fit_transform(df[categorical_feature])

        with open(f'encoders/{categorical_feature}.pkl', 'wb') as f:
            pickle.dump(le, f)

    X_categorical = df[categorical_columns]

    # Scale the continuous features
    os.makedirs('scalers', exist_ok=True)
    for continuous_feature in continuous_columns:
        scaler = StandardScaler()
        df[continuous_feature] = scaler.fit_transform(df[continuous_feature].values.reshape(-1, 1))

        with open(f'scalers/{continuous_feature}.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    # separate the target variable 'salary_in_usd'
    y = df.pop('salary_in_usd')

    X_continuous = df[continuous_columns[:-1]]

    return X_categorical, X_continuous, y


def prepare_dataframe_to_test(train_df_path: str):
    df = pd.read_csv(train_df_path)

    # drop the 'salary' column
    df.drop(columns=['salary'], inplace=True)

    # Get all categorical features
    for categorical_feature in categorical_columns:
        with open(f'encoders/{categorical_feature}.pkl', 'rb') as f:
            le = pickle.load(f)
        df[categorical_feature] = le.transform(df[categorical_feature])

    X_categorical = df[categorical_columns]

    # Scale the continuous features
    os.makedirs('scalers', exist_ok=True)
    for continuous_feature in continuous_columns:
        with open(f'scalers/{continuous_feature}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        df[continuous_feature] = scaler.transform(df[continuous_feature].values.reshape(-1, 1))

    # separate the target variable 'salary_in_usd'
    y = df.pop('salary_in_usd')

    X_continuous = df[continuous_columns[:-1]]

    return X_categorical, X_continuous, y


if __name__ == '__main__':
    X_categorical, X_continuous, y = prepare_dataframe_to_train("train.csv")
    os.makedirs('train_data', exist_ok=True)
    with open(f'train_data/X_categorical.pkl', 'wb') as f:
        pickle.dump(X_categorical, f)
    with open(f'train_data/X_continuous.pkl', 'wb') as f:
        pickle.dump(X_continuous, f)
    with open(f'train_data/y.pkl', 'wb') as f:
        pickle.dump(y, f)

    X_categorical, X_continuous, y = prepare_dataframe_to_test("test.csv")
    os.makedirs('test_data', exist_ok=True)
    with open(f'test_data/X_categorical.pkl', 'wb') as f:
        pickle.dump(X_categorical, f)
    with open(f'test_data/X_continuous.pkl', 'wb') as f:
        pickle.dump(X_continuous, f)
    with open(f'test_data/y.pkl', 'wb') as f:
        pickle.dump(y, f)