"""
Module: preprocessing.py
Single Responsibility Principle (SRP):
Data ingestion, missing value imputation, domain feature engineering,
categorical encoding, and stratified splitting.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def load_raw_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Step 1: Ingest raw CSV datasets.
    """
    # TODO Step 1.1: Use pd.read_csv() to load train_path and test_path
    raw_train_df = pd.read_csv(train_path)
    raw_test_df = pd.read_csv(test_path)

    # TODO Step 1.2: Check that 'Survived' column exists in train_df
    if 'Survived' not in raw_train_df.columns:
        raise ValueError("'Survived' column missing from train_df")
    if 'Survived' in raw_test_df.columns:
        raise ValueError("'Survived' column should not be in test_df")

    # TODO Step 1.3: Safely cache test_df['PassengerId'] for final submissions
    passenger_ids = raw_test_df['PassengerId'].copy()

    # TODO Step 1.4: Return the DataFrames and passenger_ids
    return raw_train_df, raw_test_df, passenger_ids


def drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Remove uninformative identifiers and overly sparse attributes.
    """
    # TODO Step 2.1: Define list of uninformative columns to drop
    columns_to_drop = ['PassengerId', 'Ticket', 'Cabin']

    # TODO Step 2.2: Use df.drop(columns=..., errors='ignore') to drop safely
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # TODO Step 2.3: Return the cleaned DataFrame
    return df


def impute_missing_values(
    train_df: pd.DataFrame, 
    test_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Step 3: Missingness imputation.
    - Impute continuous numerical columns (Age, Fare) with median.
    - Impute categorical nominal column (Embarked) with mode ('S').
    - Fit on train_df ONLY to prevent data leakage, then transform test_df.
    """
    # TODO Step 3.1: Calculate median Age and Fare from train_df
    num_imputer = SimpleImputer(strategy='median')
    train_df[["Age", "Fare"]] = num_imputer.fit_transform(train_df[["Age", "Fare"]])

    # TODO Step 3.2: Impute categorical Embarked with mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    train_df[["Embarked"]] = cat_imputer.fit_transform(train_df[["Embarked"]])

    # TODO Step 3.3: Transform test_df using the fitted train imputers (no fit_transform on test!)
    if test_df is not None:
        test_df[["Age", "Fare"]] = num_imputer.transform(test_df[["Age", "Fare"]])
        test_df[["Embarked"]] = cat_imputer.transform(test_df[["Embarked"]])

    # TODO Step 3.4: Verify zero null values remain
    assert train_df[["Age", "Fare", "Embarked"]].isnull().sum().sum() == 0
    if test_df is not None:
        assert test_df[["Age", "Fare", "Embarked"]].isnull().sum().sum() == 0

    # TODO Step 3.5: Return imputed (train_df, test_df)
    return train_df, test_df


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Domain feature engineering.
    - FamilySize = SibSp + Parch + 1
    - IsAlone = 1 if FamilySize == 1 else 0
    - Title = Extracted prefix from Name mapped to: Mr, Miss, Mrs, Master, Rare
    """
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss', 'Mlle': 'Miss', 'Ms': 'Miss',
        'Mrs': 'Mrs', 'Mme': 'Mrs',
        'Master': 'Master'
    }

    # TODO Step 4.1: Compute FamilySize and IsAlone
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # TODO Step 4.2: Extract Title from Name regex and map rare titles
    if 'Name' in df.columns:
        raw_titles = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = raw_titles.map(lambda x: title_mapping.get(x, 'Rare'))
        df = df.drop(columns=['Name'], errors='ignore')

    # TODO Step 4.3: Return the enriched DataFrame
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Encode categoricals into numeric representations.
    - Binary encode 'Sex': {'male': 0, 'female': 1}
    - One-hot encode nominal columns ('Embarked', 'Title') with drop_first=True
    """
    # TODO Step 5.1: Safely map 'Sex' column to 0 (male) and 1 (female)
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).fillna(df['Sex']).astype(int)

    # TODO Step 5.2: Use pd.get_dummies with drop_first=True
    cat_to_encode = [col for col in ['Embarked', 'Title'] if col in df.columns]
    df = pd.get_dummies(df, columns=cat_to_encode, drop_first=True, dtype=int)

    # TODO Step 5.3: Verify all column dtypes in the DataFrame are numeric
    assert df.select_dtypes(include=[np.number]).shape[1] == df.shape[1], "Non-numeric columns remain!"

    # TODO Step 5.4: Return the encoded DataFrame
    return df


def perform_stratified_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Step 6: Stratified train/val split (80/20) preserving ~38.4% survival ratio.
    """
    # TODO Step 6.1: Execute train_test_split with stratify=y
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # TODO Step 6.2: Print validation check verifying shapes and survival rates
    print(f"[*] Split complete: X_train {X_train.shape} | X_val {X_val.shape}")
    print(f"    Train survival rate: {y_train.mean():.3f} | Val survival rate: {y_val.mean():.3f}")

    # TODO Step 6.3: Return (X_train, X_val, y_train, y_val)
    return X_train, X_val, y_train, y_val


def run_full_preprocessing_pipeline(
    train_path: str, 
    test_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Master Preprocessing Function: Orchestrates Steps 1 through 6.
    """
    # TODO Step 7.1: Load raw data via load_raw_data()
    raw_train_df, raw_test_df, passenger_ids = load_raw_data(train_path, test_path)

    # TODO Step 7.2: Drop identifier columns via drop_uninformative_columns()
    train_df = drop_uninformative_columns(raw_train_df)
    test_df = drop_uninformative_columns(raw_test_df) if raw_test_df is not None else None

    # TODO Step 7.3: Impute missing values via impute_missing_values()
    train_df, test_df = impute_missing_values(train_df, test_df)

    # TODO Step 7.4: Engineer domain features via engineer_domain_features()
    train_df = engineer_domain_features(train_df)
    if test_df is not None:
        test_df = engineer_domain_features(test_df)

    # TODO Step 7.5: Encode categoricals via encode_categorical_features()
    train_df = encode_categorical_features(train_df)
    if test_df is not None:
        test_df = encode_categorical_features(test_df)

    # TODO Step 7.6: Separate features X and label y ('Survived')
    X = train_df.drop('Survived', axis=1)
    y = train_df['Survived']

    # TODO Step 7.7: Call perform_stratified_split(X, y) and return splits
    X_train, X_val, y_train, y_val = perform_stratified_split(X, y)

    return X_train, X_val, y_train, y_val, passenger_ids
