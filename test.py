import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_concatenated_text(df: pd.DataFrame, target_column: str | None = None) -> pd.Series:
    """
    Concatenate all non-target columns into a single text string per row.
    This treats structured fields as tokens so a single TF-IDF can be learned.
    """
    columns_to_use = [c for c in df.columns if c != target_column] if target_column else list(df.columns)
    return df[columns_to_use].astype(str).apply(lambda row: " | ".join(row.values).lower(), axis=1)


if __name__ == "__main__":
    print("Loading data...")
    try:
        train_df = pd.read_csv("kaggle_train_dataset.csv")
        test_df = pd.read_csv("kaggle_test_dataset.csv")
        print(f"Training data loaded: {train_df.shape[0]} records.")
        print(f"Test data loaded: {test_df.shape[0]} records.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure 'kaggle_train_dataset.csv' and 'kaggle_test_dataset.csv' are in the same directory as this script.")
        raise SystemExit(1)

    if "ASA_Rating" not in train_df.columns:
        raise SystemExit("Expected target column 'ASA_Rating' in training data.")

    print("Building text features...")
    train_text = build_concatenated_text(train_df, target_column="ASA_Rating")
    test_text = build_concatenated_text(test_df)

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    y_train = train_df["ASA_Rating"].astype(int)

    print("Training classifier (Logistic Regression)...")
    model = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    print("Predicting test ASA_Rating...")
    y_pred = model.predict(X_test)

    print("Writing submission file...")
    submission_df = pd.DataFrame({
        "Id": range(len(test_df)),
        "ASA_Rating": y_pred
    })
    submission_df.to_csv("kaggle_submission.csv", index=False)
    print("Submission file 'kaggle_submission.csv' has been created successfully.")
