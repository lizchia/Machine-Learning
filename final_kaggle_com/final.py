from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.utils.class_weight import compute_sample_weight


def load_training_data(filename: str = "kaggle_train_dataset.csv") -> pd.DataFrame:
    """
    Load the Kaggle training dataset that lives in the same directory
    as this script (e.g., in Google Drive).
    """
    script_dir = Path(__file__).resolve().parent
    dataset_path = script_dir / filename
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    return pd.read_csv(dataset_path)


def select_columns(
    df: pd.DataFrame,
    columns: Sequence[str] = (
        "ASA_Rating",
        "Gender",
        "ICU_Patient",
        "Age",
        "HEIGHT",
        "WEIGHT",
        "Patient_Source",
        "Anesthesia_Method",
        "Lab_Values",
        "Medication_Usage",
        "properties_display",
        "Catheter_Use",
    ),
) -> pd.DataFrame:
    """Return a copy of the dataframe restricted to the specified columns."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    return df.loc[:, columns].copy()


FEATURE_COLUMNS: list[str] = [
    "Gender",
    "ICU_Patient",
    "Age",
    "HEIGHT_cm",
    "WEIGHT",
    "Patient_Source",
    "Anesthesia_Method",
    # Engineered clinical features
    "num_lab_abnormal_high",
    "num_lab_abnormal_low",
    "num_lab_abnormal_total",
    "anemia_flag",
    "renal_impairment_flag",
    "electrolyte_imbalance_flag",
    "diabetes_med_flag",
    "antihypertensive_flag",
    "anticoagulant_flag",
    "opioid_flag",
    "has_urinary_catheter",
    "num_catheters",
]


def encode_binary_column(
    series: pd.Series, true_value: str, false_value: str | None = None
) -> pd.Series:
    """
    Convert a column with two categories into 0/1.

    true_value is mapped to 1. If false_value is provided, it is mapped to 0.
    All other values raise a ValueError to prevent silent issues.
    """
    mapping = {true_value: 1}
    if false_value is not None:
        mapping[false_value] = 0
    encoded = series.map(mapping)
    if encoded.isna().any():
        unknown = series[encoded.isna()].unique()
        raise ValueError(f"Unexpected values for binary encoding: {unknown}")
    return encoded.astype("int8")


def label_encode(series: pd.Series) -> tuple[pd.Series, pd.Index]:
    """
    Apply pandas factorize to perform label encoding, returning the
    encoded series and the categories used.
    """
    stringified = series.astype("string")
    cleaned = stringified.str.strip()
    codes, uniques = pd.factorize(cleaned, sort=True)
    encoded = pd.Series(codes, index=series.index).astype("int16")
    return encoded, uniques


def _parse_height_to_inches(value: str | float | int) -> float:
    """
    Convert a single height value to inches.

    Handles:
    - plain numeric inches (e.g. 63, "63", "63.0")
    - feet and inches strings like "5' 3", "5'3", "5'3\"", "5 ft 3 in"
    """
    if pd.isna(value):
        return float("nan")

    # Already numeric → assume inches
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    # Try direct float first
    try:
        return float(s)
    except ValueError:
        pass

    # Normalize common characters
    s = s.lower().replace("ft", "").replace("feet", "").replace("in", "").replace('"', "")

    if "'" in s:
        feet_part, inches_part = s.split("'", 1)
        feet_part = feet_part.strip()
        inches_part = inches_part.strip()
        feet = float(feet_part) if feet_part else 0.0
        inches = float(inches_part) if inches_part else 0.0
        return feet * 12.0 + inches

    # If nothing worked, raise to expose unexpected format
    raise ValueError(f"Unrecognized height format: {value!r}")


def convert_height_to_cm(series: pd.Series) -> pd.Series:
    """Convert height values (inches, or feet'inches strings) to centimeters."""
    inches = series.apply(_parse_height_to_inches)
    return inches * 2.54


def add_clinical_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive simple clinically meaningful features from long text fields:
    Lab_Values, Medication_Usage, properties_display, Catheter_Use.
    """
    # Work on a copy to avoid side effects if reused, but return the same reference
    # for chaining convenience.
    if "Lab_Values" in df.columns:
        lab = df["Lab_Values"].astype("string").fillna("")
        lab_upper = lab.str.upper()
        # Count abnormal highs and lows
        df["num_lab_abnormal_high"] = lab.str.count(r"\(H\)").astype("int16")
        df["num_lab_abnormal_low"] = lab.str.count(r"\(L\)").astype("int16")
        df["num_lab_abnormal_total"] = (
            df["num_lab_abnormal_high"] + df["num_lab_abnormal_low"]
        ).astype("int16")
        # Simple condition flags
        df["anemia_flag"] = (
            lab_upper.str.contains("HEMOGLOBIN") & lab.str.contains(r"\(L\)")
            | lab_upper.str.contains("HEMATOCRIT") & lab.str.contains(r"\(L\)")
        ).astype("int8")
        df["renal_impairment_flag"] = (
            lab_upper.str.contains("CREATININE") & lab.str.contains(r"\(H\)")
        ).astype("int8")
        df["electrolyte_imbalance_flag"] = (
            lab_upper.str.contains("SODIUM|POTASSIUM|CALCIUM")
            & lab.str.contains(r"\((H|L)\)")
        ).astype("int8")

    if "Medication_Usage" in df.columns:
        meds = df["Medication_Usage"].astype("string").fillna("").str.upper()
        df["diabetes_med_flag"] = meds.str.contains(
            "INSULIN|METFORMIN|GLIPIZIDE|GLYBURIDE|SITAGLIPTIN"
        ).astype("int8")
        df["antihypertensive_flag"] = meds.str.contains(
            "AMLODIPINE|LOSARTAN|LISINOPRIL|METOPROLOL|ATENOLOL|BETA BLOCKER|ACE INHIBITOR|ARB"
        ).astype("int8")
        df["anticoagulant_flag"] = meds.str.contains(
            "WARFARIN|HEPARIN|ENOXAPARIN|XARELTO|APIXABAN|RIVAROXABAN|ASPIRIN|CLOPIDOGREL"
        ).astype("int8")
        df["opioid_flag"] = meds.str.contains(
            "MORPHINE|FENTANYL|HYDROMORPHONE|OXYCODONE|HYDROCODONE"
        ).astype("int8")

    if "Catheter_Use" in df.columns:
        cath = df["Catheter_Use"].astype("string").fillna("").str.upper()
        df["has_urinary_catheter"] = cath.str.contains("URINARY CATHETER").astype("int8")
        df["num_catheters"] = (
            cath.apply(lambda s: len([part for part in s.split(",") if part.strip()]))
        ).astype("int16")

    # Ensure all engineered columns exist even if source columns were missing
    for col in [
        "num_lab_abnormal_high",
        "num_lab_abnormal_low",
        "num_lab_abnormal_total",
        "anemia_flag",
        "renal_impairment_flag",
        "electrolyte_imbalance_flag",
        "diabetes_med_flag",
        "antihypertensive_flag",
        "anticoagulant_flag",
        "opioid_flag",
        "has_urinary_catheter",
        "num_catheters",
    ]:
        if col not in df.columns:
            df[col] = 0
    return df


def transform_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Select the requested columns and apply the specified transformations.
    """
    working = select_columns(df)
    working["Patient_Source"] = encode_binary_column(
        working["Patient_Source"], true_value="Inpatient", false_value="Outpatient"
    )
    working["ICU_Patient"] = encode_binary_column(
        working["ICU_Patient"], true_value="Yes", false_value="No"
    )
    height_cm = convert_height_to_cm(working["HEIGHT"])
    mean_height_cm = height_cm.mean(skipna=True)
    height_cm = height_cm.fillna(mean_height_cm)
    working = working.drop(columns=["HEIGHT"])
    working["HEIGHT_cm"] = height_cm
    numeric_means: dict[str, float] = {}
    for column in ["Age", "WEIGHT"]:
        numeric_series = working[column].astype(float)
        mean_value = numeric_series.mean(skipna=True)
        numeric_means[column] = mean_value
        working[column] = numeric_series.fillna(mean_value)
    categorical_fill_values: dict[str, str] = {}
    for column in ["Anesthesia_Method", "Gender"]:
        string_series = working[column].astype("string").str.strip()
        non_null = string_series.dropna()
        fill_value = non_null.mode().iloc[0] if not non_null.empty else ""
        categorical_fill_values[column] = fill_value
        working[column] = string_series.fillna(fill_value)

    working["Anesthesia_Method"], anesthesia_categories = label_encode(
        working["Anesthesia_Method"]
    )
    working["Gender"], gender_categories = label_encode(working["Gender"])
    scalers: dict[str, dict[str, float]] = {}
    for col in ["Age", "WEIGHT", "HEIGHT_cm"]:
        scaled, params = min_max_scale(working[col].astype(float))
        working[col] = scaled
        scalers[col] = params

    # Add engineered clinical features from long text columns
    working = add_clinical_text_features(working)
    metadata = {
        "categories": {
            "Anesthesia_Method": anesthesia_categories,
            "Gender": gender_categories,
        },
        "height_mean_cm": mean_height_cm,
        "numeric_means": numeric_means,
        "scalers": scalers,
        "category_fill_values": categorical_fill_values,
    }
    return working, metadata


def count_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return the number of missing values for every column."""
    return df.isna().sum()


def apply_label_mapping(
    series: pd.Series, categories: pd.Index, fallback_value: str | None = None
) -> pd.Series:
    """
    Map categorical strings to the integer labels learned during training.
    """
    mapping = {str(value): idx for idx, value in enumerate(categories)}
    cleaned = series.astype("string").str.strip()
    encoded = cleaned.map(mapping)
    if encoded.isna().any():
        if fallback_value is not None:
            fallback_key = str(fallback_value)
            if fallback_key not in mapping:
                raise ValueError(
                    f"Fallback value {fallback_value!r} not present in category mapping."
                )
            encoded = encoded.fillna(mapping[fallback_key])
        else:
            unknown = cleaned[encoded.isna()].unique()
            raise ValueError(
                f"Encountered unseen category values {unknown} for column being encoded."
            )
    return encoded.astype("int16")


def min_max_scale(
    series: pd.Series, params: dict[str, float] | None = None
) -> tuple[pd.Series, dict[str, float]] | pd.Series:
    """
    Apply min-max scaling. When params is None, fit on the series and return
    both the scaled series and the fitted parameters. When params are provided,
    only the scaled series is returned.
    """
    if params is None:
        min_val = series.min()
        max_val = series.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            fitted = {"min": float(min_val), "max": float(max_val)}
            return series, fitted
        scaled = (series - min_val) / (max_val - min_val)
        fitted = {"min": float(min_val), "max": float(max_val)}
        return scaled, fitted

    min_val = params["min"]
    max_val = params["max"]
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


FEATURE_COLUMNS: list[str] = [
    "Gender",
    "ICU_Patient",
    "Age",
    "HEIGHT_cm",
    "WEIGHT",
    "Patient_Source",
    "Anesthesia_Method",
]


def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """
    Perform hyperparameter tuning for a Random Forest classifier using RandomizedSearchCV.
    """
    param_distributions = {
        "n_estimators": [200, 300, 400, 500, 600],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["sqrt", "log2", 0.6, 0.8],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_distributions=param_distributions,
        n_iter=20,
        scoring="accuracy",
        cv=3,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_random_forest(
    df: pd.DataFrame, *, tune: bool = True
) -> tuple[RandomForestClassifier, float, str, dict[str, Any]]:
    """
    Train a Random Forest classifier using the specified feature set and ASA_Rating label.
    Optionally perform hyperparameter tuning before final evaluation.
    """
    target_column = "ASA_Rating"
    required_columns = FEATURE_COLUMNS + [target_column]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for training: {missing}")

    clean_df = df.dropna(subset=required_columns)
    if clean_df.empty:
        raise ValueError("No rows available for training after dropping missing values.")

    X = clean_df[FEATURE_COLUMNS]
    y = clean_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if tune:
        model, best_params = tune_random_forest(X_train, y_train)
    else:
        model = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
        model.fit(X_train, y_train)
        best_params = model.get_params()

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    return model, accuracy, report, best_params


def tune_hist_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> tuple[HistGradientBoostingClassifier, dict[str, Any]]:
    """
    Hyperparameter tuning for HistGradientBoostingClassifier using RandomizedSearchCV.
    """
    param_distributions = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [None, 6, 10, 16],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [5, 10, 20],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0],
        "max_bins": [64, 128, 255],
        "early_stopping": [False],
    }
    base_model = HistGradientBoostingClassifier(
        random_state=random_state, loss="log_loss", max_iter=500
    )
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="accuracy",
        cv=3,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    search.fit(X_train, y_train, sample_weight=sample_weight)
    return search.best_estimator_, search.best_params_


def train_hist_gradient_boosting(
    df: pd.DataFrame, *, tune: bool = True
) -> tuple[HistGradientBoostingClassifier, float, str, dict[str, Any]]:
    """
    Train a HistGradientBoostingClassifier on the transformed dataframe.
    """
    target_column = "ASA_Rating"
    required_columns = FEATURE_COLUMNS + [target_column]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for training: {missing}")

    clean_df = df.dropna(subset=required_columns)
    if clean_df.empty:
        raise ValueError("No rows available for training after dropping missing values.")

    X = clean_df[FEATURE_COLUMNS]
    y = clean_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if tune:
        model, best_params = tune_hist_gradient_boosting(X_train, y_train)
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=0.1,
            max_bins=255,
            random_state=42,
            loss="log_loss",
            max_iter=500,
            early_stopping=False,
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        best_params = model.get_params()

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    return model, accuracy, report, best_params


def prepare_inference_features(
    df: pd.DataFrame, metadata: dict[str, Any]
) -> pd.DataFrame:
    """
    Apply the same preprocessing steps to the inference dataframe.
    """
    working = select_columns(
        df,
        (
            "Gender",
            "ICU_Patient",
            "Age",
            "HEIGHT",
            "WEIGHT",
            "Patient_Source",
            "Anesthesia_Method",
            "Lab_Values",
            "Medication_Usage",
            "properties_display",
            "Catheter_Use",
        ),
    )
    working["Patient_Source"] = encode_binary_column(
        working["Patient_Source"], true_value="Inpatient", false_value="Outpatient"
    )
    working["ICU_Patient"] = encode_binary_column(
        working["ICU_Patient"], true_value="Yes", false_value="No"
    )
    height_cm = convert_height_to_cm(working["HEIGHT"])
    height_cm = height_cm.fillna(metadata["height_mean_cm"])
    working = working.drop(columns=["HEIGHT"])
    working["HEIGHT_cm"] = height_cm

    for col in ["Age", "WEIGHT"]:
        mean_val = metadata["numeric_means"][col]
        working[col] = working[col].astype(float).fillna(mean_val)

    categories = metadata["categories"]
    fill_values = metadata["category_fill_values"]
    working["Anesthesia_Method"] = apply_label_mapping(
        working["Anesthesia_Method"],
        categories["Anesthesia_Method"],
        fallback_value=fill_values["Anesthesia_Method"],
    )
    working["Gender"] = apply_label_mapping(
        working["Gender"],
        categories["Gender"],
        fallback_value=fill_values["Gender"],
    )

    for col in ["Age", "WEIGHT", "HEIGHT_cm"]:
        params = metadata["scalers"][col]
        scaled = min_max_scale(working[col].astype(float), params=params)
        working[col] = scaled

    # Apply same clinical text feature extraction
    working = add_clinical_text_features(working)

    return working.loc[:, FEATURE_COLUMNS]


def load_test_data(filename: str = "kaggle_test_dataset.csv") -> pd.DataFrame:
    """Load and return the Kaggle test dataset."""
    script_dir = Path(__file__).resolve().parent
    dataset_path = script_dir / filename
    if not dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found at: {dataset_path}")
    df = pd.read_csv(dataset_path)
    return df.reset_index(drop=True)


def save_submission(
    ids: Sequence[int] | pd.Series,
    predictions: Sequence[int],
    output_name: str = "kaggle_submission.csv",
) -> Path:
    """Save the predictions to a Kaggle submission-style CSV."""
    submission = pd.DataFrame({"Id": ids, "ASA_Rating": predictions})
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / output_name
    submission.to_csv(output_path, index=False)
    return output_path

def load_test_features(filename: str = "kaggle_test_dataset.csv") -> pd.DataFrame:
    """
    Load the Kaggle test dataset and return only the requested feature columns.
    """
    script_dir = Path(__file__).resolve().parent
    dataset_path = script_dir / filename
    if not dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found at: {dataset_path}")
    test_df = pd.read_csv(dataset_path)
    return select_columns(
        test_df,
        (
            "Gender",
            "ICU_Patient",
            "Age",
            "HEIGHT",
            "WEIGHT",
            "Patient_Source",
            "Anesthesia_Method",
        ),
    )


if __name__ == "__main__":
    training_df = load_training_data()
    transformed_df, metadata = transform_dataset(training_df)
    print("Transformed preview:")
    print(transformed_df.head())
    for column, categories in metadata["categories"].items():
        print(f"\n{column} label mapping (code -> label):")
        for code, label in enumerate(categories):
            print(f"{code}: {label}")
    print("\nMissing values per column (transformed dataset):")
    print(count_missing_values(transformed_df))
    rf_model, rf_accuracy, rf_report, rf_best_params = train_random_forest(
        transformed_df
    )
    print("\nRandom Forest accuracy:", rf_accuracy)
    print("\nRandom Forest classification report:")
    print(rf_report)
    print("\nRandom Forest best hyperparameters:", rf_best_params)

    hgb_model, hgb_accuracy, hgb_report, hgb_best_params = train_hist_gradient_boosting(
        transformed_df
    )
    print("\nHistGradientBoosting accuracy:", hgb_accuracy)
    print("\nHistGradientBoosting classification report:")
    print(hgb_report)
    print("\nHistGradientBoosting best hyperparameters:", hgb_best_params)

    best_model, best_name, best_accuracy = (
        (hgb_model, "HistGradientBoosting", hgb_accuracy)
        if hgb_accuracy >= rf_accuracy
        else (rf_model, "RandomForest", rf_accuracy)
    )
    print(f"\nUsing {best_name} for test predictions (accuracy={best_accuracy:.3f}).")

    test_df = load_test_data()
    test_features = prepare_inference_features(test_df, metadata)
    predictions = best_model.predict(test_features)
    submission_path = save_submission(test_df.index, predictions)
    print(f"\nSaved predictions to {submission_path}")
