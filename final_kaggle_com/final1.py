import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import re
# Added train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score # Added for detailed reporting

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
    """Robust CSV loader."""
    encodings = ['utf-8', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, engine='python', on_bad_lines='skip')
            return df
        except:
            continue
    raise ValueError(f"Could not load {filepath}")

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

def extract_lab_value(text, lab_name):
    """Regex to find specific lab number (e.g. 'Creatinine: 0.5')"""
    if pd.isna(text): return np.nan
    pattern = re.compile(re.escape(lab_name) + r"[:\s]+([\d\.]+)", re.IGNORECASE)
    match = pattern.search(str(text))
    if match:
        try:
            return float(match.group(1))
        except:
            return np.nan
    return np.nan

def clean_dataframe(df):
    df = df.copy()
    
    # 1. Basic Cleaning
    if 'HEIGHT' in df.columns:
        df['HEIGHT_clean'] = df['HEIGHT'].apply(parse_height)
    else:
        df['HEIGHT_clean'] = np.nan
    df['WEIGHT_clean'] = pd.to_numeric(df['WEIGHT'], errors='coerce')
    
    # 2. BMI Calculation
    height_m = df['HEIGHT_clean'] * 0.0254
    df['BMI'] = df['WEIGHT_clean'] / (height_m ** 2)
    df['BMI'] = df['BMI'].replace([np.inf, -np.inf], np.nan)

    # 3. Elderly Flag
    df['Is_Elderly'] = (df['Age'] >= 65).astype(int)

    # 4. Extract Specific Lab Values
    labs_to_extract = ['Creatinine', 'Glucose', 'Hemoglobin', 'Potassium', 'Sodium', 'Urea nitrogen']
    for lab in labs_to_extract:
        df[f'Lab_{lab}'] = df['Lab_Values'].apply(lambda x: extract_lab_value(x, lab))

    # 5. Lab Abnormal Count
    df['Lab_Abnormal_Count'] = df['Lab_Values'].astype(str).apply(lambda x: x.count('(H)') + x.count('(L)'))
    
    # 6. Combined Text for TF-IDF
    text_cols = ['Surgery_Name', 'Medication_Usage', 'Lab_Values', 'Catheter_Use']
    for col in text_cols:
        if col in df.columns: df[col] = df[col].fillna('')
        else: df[col] = ''
    
    df['combined_text'] = (df['Surgery_Name'] + " " + df['Medication_Usage'] + " " + 
                           df['Lab_Values'] + " " + df['Catheter_Use'])
    return df

print("Preprocessing: Extracting specific lab values and BMI...")
train_clean = clean_dataframe(train_df)
test_clean = clean_dataframe(test_df)

X = train_clean
y = train_clean['ASA_Rating']
X_test = test_clean

# ==========================================
# 3. SPLIT TRAINING DATA (Local Validation)
# ==========================================
# We split the training data 80/20.
# 80% is used to train, 20% is used to "validate" (simulate the test).
print("Splitting 'kaggle_train_dataset' into Training (80%) and Local Validation (20%)...")
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 4. Pipeline Construction
# ==========================================

numeric_features = [
    'Age', 'HEIGHT_clean', 'WEIGHT_clean', 'BMI', 'Lab_Abnormal_Count',
    'Lab_Creatinine', 'Lab_Glucose', 'Lab_Hemoglobin', 'Lab_Potassium', 'Lab_Sodium', 'Lab_Urea nitrogen'
]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

categorical_features = ['Gender', 'ICU_Patient', 'Anesthesia_Method', 'Patient_Source', 'Is_Elderly']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

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
# 5. Models
# ==========================================

models = {
    "Method #1 (Naive Bayes)": MultinomialNB(alpha=0.01),
    "Method #2 (Logistic Regression)": LogisticRegression(max_iter=3000, C=1.5, solver='liblinear', class_weight='balanced'),
    "Method #3 (Decision Tree)": DecisionTreeClassifier(max_depth=20, min_samples_leaf=15, class_weight='balanced', random_state=42),
    "Method #4 (KNN)": KNeighborsClassifier(n_neighbors=19, weights='distance', metric='cosine'),
    "Method #5 (SVM)": LinearSVC(C=0.3, class_weight='balanced', dual=False, random_state=42, max_iter=3000)
}

# ==========================================
# 6. Training, Evaluation & Submission
# ==========================================

print("\n" + "="*60)
print("STARTING EXPERIMENTS WITH VALIDATION SPLIT")
print("="*60)

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
    
    #  Local Validation Split
    clf.fit(X_train_split, y_train_split)
    val_preds = clf.predict(X_val_split)
    val_f1 = f1_score(y_val_split, val_preds, average='macro')
    
    print(f"   [Local Validation] Macro F1 Score (on 20% holdout): {val_f1:.4f}")
    
    #  Cross Validation 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'f1_macro': 'f1_macro', 'prec_macro': 'precision_macro', 'rec_macro': 'recall_macro',
        'f1_micro': 'f1_micro', 'prec_micro': 'precision_micro', 'rec_micro': 'recall_micro'
    }
    
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
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
    print(f"   Retraining {name} on FULL dataset for Kaggle Submission...")
    clf.fit(X, y) # Train on ALL data (Train + Validation) for best results
    predictions = clf.predict(X_test)
    
    short_name = name.split('(')[1].split(')')[0].replace(" ", "")
    filename = f"kaggle_submission_{short_name}.csv"
    
    if len(submission_ids) != len(predictions):
        submission_ids = range(0, len(predictions))

    submission_df = pd.DataFrame({'Id': submission_ids, 'ASA_Rating': predictions})
    submission_df.to_csv(filename, index=False)
    print(f"   Saved submission to: {filename}")

print("\n" + "="*60)
print("ALL DONE!")