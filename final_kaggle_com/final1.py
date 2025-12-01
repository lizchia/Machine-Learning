import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import the 5 requested models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC 

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
TRAIN_PATH = 'kaggle_train_dataset.csv'
TEST_PATH = 'kaggle_test_dataset.csv'
SUBMISSION_TEMPLATE = 'kaggle_test_submission.csv'


print("Loading data...")

def load_data_robust(filepath):
    """
    Robust CSV loader to handle 'EOF inside string' and encoding issues.
    """
    encodings = ['utf-8', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            # on_bad_lines='skip' ensures one bad row doesn't crash the script
            df = pd.read_csv(filepath, encoding=enc, engine='python', on_bad_lines='skip')
            return df
        except Exception as e:
            try:
                # Fallback for older pandas versions
                df = pd.read_csv(filepath, encoding=enc, engine='python', error_bad_lines=False)
                return df
            except:
                continue
    raise ValueError(f"Could not load {filepath}. Check for corruption.")

try:
    train_df = load_data_robust(TRAIN_PATH)
    test_df = load_data_robust(TEST_PATH)
    print(f" -> Loaded {len(train_df)} training rows and {len(test_df)} test rows.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

# ==========================================
# 2. Preprocessing Helpers
# ==========================================

def parse_height(height_str):
    if pd.isna(height_str): return np.nan
    height_str = str(height_str).strip()
    match_ft = re.search(r"(\d+)'\s*(\d+)?", height_str)
    if match_ft:
        feet = int(match_ft.group(1))
        inches = int(match_ft.group(2)) if match_ft.group(2) else 0
        return (feet * 12) + inches
    try:
        val = float(height_str)
        if val > 100: return val * 0.393701
        return val
    except ValueError:
        return np.nan

def clean_dataframe(df):
    df = df.copy()
    if 'HEIGHT' in df.columns:
        df['HEIGHT_clean'] = df['HEIGHT'].apply(parse_height)
    else:
        df['HEIGHT_clean'] = np.nan
        
    df['WEIGHT_clean'] = pd.to_numeric(df['WEIGHT'], errors='coerce')
    
    text_cols = ['Surgery_Name', 'Medication_Usage', 'Lab_Values', 'Catheter_Use']
    for col in text_cols:
        if col in df.columns: df[col] = df[col].fillna('')
        else: df[col] = ''
    
    df['combined_text'] = (df['Surgery_Name'] + " " + df['Medication_Usage'] + " " + 
                           df['Lab_Values'] + " " + df['Catheter_Use'])
    return df

print("Preprocessing data...")
train_clean = clean_dataframe(train_df)
test_clean = clean_dataframe(test_df)

X = train_clean
y = train_clean['ASA_Rating']
X_test = test_clean

# ==========================================
# 3. Pipeline Construction
# ==========================================

numeric_features = ['Age', 'HEIGHT_clean', 'WEIGHT_clean']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler()) 
])

categorical_features = ['Gender', 'ICU_Patient', 'Anesthesia_Method', 'Patient_Source']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Increased max_features to 3000 to capture more medical terms
text_features = 'combined_text'
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2)))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_features)
    ])

# ==========================================
# 4. TUNED Experiment Setup (5 Methods)
# ==========================================

models = {
    # Alpha 0.01: Very low smoothing lets it catch rare medical terms aggressively
    "Method #1 (Naive Bayes)": MultinomialNB(alpha=0.01),
    
    # Class Weight Balanced: Crucial for fixing the imbalance between Ratings 1-4
    "Method #2 (Logistic Regression)": LogisticRegression(
        max_iter=3000, C=1.5, solver='liblinear', class_weight='balanced'
    ),
    
    # Min Samples Leaf 10: Prevents "over-memorizing" single patients
    "Method #3 (Decision Tree)": DecisionTreeClassifier(
        max_depth=20, min_samples_leaf=10, class_weight='balanced', random_state=42
    ),
    
    # Metric Cosine: Much better for Text (TF-IDF) data than standard distance
    "Method #4 (KNN)": KNeighborsClassifier(
        n_neighbors=19, weights='distance', metric='cosine'
    ),
    
    # C=0.2: Stronger regularization prevents overfitting on the text data
    "Method #5 (SVM)": LinearSVC(
        C=0.2, class_weight='balanced', dual=False, random_state=42, max_iter=3000
    )
}

# ==========================================
# 5. Training, Evaluation & Submission
# ==========================================

print("\n" + "="*60)
print("STARTING TUNED EXPERIMENTS")
print("="*60)

# ID Handling
submission_ids = None
if os.path.exists(SUBMISSION_TEMPLATE):
    try:
        sub_temp = pd.read_csv(SUBMISSION_TEMPLATE)
        submission_ids = sub_temp['Id']
    except: pass
if submission_ids is None:
    submission_ids = range(0, len(test_clean))

for name, model in models.items():
    print(f"\nRunning {name}...")
    
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'f1_macro': 'f1_macro', 'prec_macro': 'precision_macro', 'rec_macro': 'recall_macro',
        'f1_micro': 'f1_micro', 'prec_micro': 'precision_micro', 'rec_micro': 'recall_micro'
    }
    
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    # Print Results
    print(f"--- Results for {name} ---")
    print(">> COPY THESE VALUES TO YOUR TABLE:")
    
    micro_p = np.mean(scores['test_prec_micro'])
    micro_r = np.mean(scores['test_rec_micro'])
    micro_f1 = np.mean(scores['test_f1_micro'])
    print(f"   Average-Micro (P/R/F1): {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")
    
    macro_p = np.mean(scores['test_prec_macro'])
    macro_r = np.mean(scores['test_rec_macro'])
    macro_f1 = np.mean(scores['test_f1_macro'])
    print(f"   Average-Macro (P/R/F1): {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")
    
    # Final Training & Prediction
    print(f"   Retraining {name} on full dataset...")
    clf.fit(X, y)
    predictions = clf.predict(X_test)
    
    # Create submission file
    short_name = name.split('(')[1].split(')')[0].replace(" ", "")
    filename = f"kaggle_submission_{short_name}.csv"
    
    if len(submission_ids) != len(predictions):
        submission_ids = range(0, len(predictions))

    submission_df = pd.DataFrame({'Id': submission_ids, 'ASA_Rating': predictions})
    submission_df.to_csv(filename, index=False)
    print(f"   Saved submission to: {filename}")

print("\n" + "="*60)
print("ALL DONE! Upload the 5 generated CSV files to Kaggle.")
print("="*60)