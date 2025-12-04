import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
from datetime import datetime

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore")

# ==========================================
# LOGGING SETUP
# ==========================================
class Logger(object):
    def __init__(self):
        self.filename = f"output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        
        directory = os.path.dirname(self.filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        self.terminal = sys.stdout
        self.log = open(self.filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

print("\n" + "="*60)
print(f"EXPERIMENT RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {sys.stdout.filename}")
print("="*60)

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.decomposition import TruncatedSVD

# Import Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed.")

# ==========================================
# 1. Configuration
# ==========================================
TRAIN_PATH = 'kaggle_train_dataset.csv'
TEST_PATH = 'kaggle_test_dataset.csv'
SUBMISSION_TEMPLATE = 'kaggle_submission.csv'

VALIDATION_SPLIT_SIZE = 0.15
CV_FOLDS = 10

print(f"Configuration: Using {CV_FOLDS} Cross-Validation Folds.")
print("Loading data...")

def load_data_robust(filepath):
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
    print(f" -> Loaded {len(train_df)} training rows.")
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
    if pd.isna(text): return np.nan
    pattern = re.compile(re.escape(lab_name) + r"[:\s]+([\d\.]+)", re.IGNORECASE)
    match = pattern.search(str(text))
    if match:
        try:
            return float(match.group(1))
        except:
            return np.nan
    return np.nan

def count_medications(text):
    if pd.isna(text): return 0
    return str(text).count(':')

def clean_medical_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    noise = ['mg/dl', 'mmol/l', 'thous/mcl', 'mill/mcl', 'g/dl', '%', '(n)', 'result', 'name']
    for w in noise:
        text = text.replace(w, '')
    text = text.replace('(h)', ' high_abnormal ')
    text = text.replace('(l)', ' low_abnormal ')
    return text

def calculate_comorbidity_score(row):
    full_text = str(row.get('Surgery_Name', '')) + " " + \
                str(row.get('Medication_Usage', '')) + " " + \
                str(row.get('Patient_Source', ''))
    full_text = full_text.lower()
    
    risk_dict = {
        # === CRITICAL / EMERGENCY (Score 4) ===
        'sepsis': 4, 'septic': 4, 'shock': 4, 'rupture': 4, 
        'transplant': 4, 'craniotomy': 4, 'aneurysm': 4,
        'intracranial': 4, 'bleed': 4,
        # === SEVERE CHRONIC (Score 3) ===
        'dialysis': 3, 'esrd': 3, 'failure': 3, 
        'chf': 3, 'congestive': 3, 'cabg': 3, 'valve': 3, 
        'metastasis': 3, 'malignancy': 3, 'chemo': 3, 'radiation': 3,
        'cirrhosis': 3, 'ascites': 3,
        'stroke': 3, 'cva': 3, 'paralysis': 3,
        # === MODERATE (Score 2) ===
        'cancer': 2, 'tumor': 2, 'mass': 2,
        'copd': 2, 'asthma': 2, 'pulmonary': 2, 'pneumonia': 2,
        'diabetes': 2, 'insulin': 2, 'dm': 2, 'ketoacidosis': 2,
        'angina': 2, 'cad': 2, 'arrhythmia': 2, 'pacemaker': 2,
        'kidney': 2, 'renal': 2,
        # === MILD (Score 1) ===
        'hypertension': 1, 'htn': 1, 'pressure': 1,
        'obesity': 1, 'bmi': 1, 'apnea': 1, 
        'gerd': 1, 'reflux': 1, 'anemia': 1,
        'infection': 1, 'abscess': 1
    }
    
    score = 0
    for word, points in risk_dict.items():
        if word in full_text:
            score += points
    return score

def clean_dataframe(df):
    df = df.copy()
    
    if 'HEIGHT' in df.columns:
        df['HEIGHT_clean'] = df['HEIGHT'].apply(parse_height)
    else:
        df['HEIGHT_clean'] = np.nan
    df['WEIGHT_clean'] = pd.to_numeric(df['WEIGHT'], errors='coerce')
    
    height_m = df['HEIGHT_clean'] * 0.0254
    df['BMI'] = df['WEIGHT_clean'] / (height_m ** 2)
    df['BMI'] = df['BMI'].replace([np.inf, -np.inf], np.nan)

    df['Is_Elderly'] = (df['Age'] >= 65).astype(int)
    df['Is_Child'] = (df['Age'] <= 12).astype(int)

    labs_to_extract = ['Creatinine', 'Glucose', 'Hemoglobin', 'Potassium', 'Sodium', 'Urea nitrogen', 'Platelets', 'Chloride']
    for lab in labs_to_extract:
        df[f'Lab_{lab}'] = df['Lab_Values'].apply(lambda x: extract_lab_value(x, lab))
        
    df['Lab_BUN_Creatinine_Ratio'] = df['Lab_Urea nitrogen'] / df['Lab_Creatinine'].replace(0, np.nan)

    df['Lab_High_Count'] = df['Lab_Values'].astype(str).apply(lambda x: x.count('(H)'))
    df['Lab_Low_Count'] = df['Lab_Values'].astype(str).apply(lambda x: x.count('(L)'))
    df['Lab_Total_Abnormal'] = df['Lab_High_Count'] + df['Lab_Low_Count']

    df['Medication_Count'] = df['Medication_Usage'].apply(count_medications)
    df['Comorbidity_Score'] = df.apply(calculate_comorbidity_score, axis=1)

    df['Is_Emergency'] = (
        df['Surgery_Name'].astype(str).str.contains('Emergency', case=False) | 
        df['Patient_Source'].astype(str).str.contains('Emergency', case=False)
    ).astype(int)
    
    df['text_surgery'] = df['Surgery_Name'].fillna('').apply(clean_medical_text)
    df['text_rest'] = (
        df['Medication_Usage'].fillna('') + " " + 
        df['Lab_Values'].fillna('') + " " + 
        df['Catheter_Use'].fillna('')
    ).apply(clean_medical_text)
    
    return df

print("Preprocessing: Target Encoding, SVD & Comorbidity Scoring...")
train_clean = clean_dataframe(train_df)
test_clean = clean_dataframe(test_df)

X = train_clean
y = train_clean['ASA_Rating']
X_test = test_clean

# ==========================================
# 3. SPLIT TRAINING DATA (90/10)
# ==========================================
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X, y, test_size=VALIDATION_SPLIT_SIZE, random_state=42, stratify=y
)

# ==========================================
# 4. Pipeline Construction
# ==========================================

numeric_features = [
    'Age', 'HEIGHT_clean', 'WEIGHT_clean', 'BMI', 
    'Lab_High_Count', 'Lab_Low_Count', 'Lab_Total_Abnormal', 'Medication_Count', 'Comorbidity_Score',
    'Lab_Creatinine', 'Lab_Glucose', 'Lab_Hemoglobin', 'Lab_Potassium', 'Lab_Sodium', 'Lab_Urea nitrogen',
    'Lab_BUN_Creatinine_Ratio'
]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)), 
    ('scaler', MinMaxScaler()),
    ('selector', SelectKBest(f_classif, k=50))
])

categorical_features = ['Gender', 'ICU_Patient', 'Anesthesia_Method', 'Patient_Source', 'Is_Elderly', 'Is_Child', 'Is_Emergency']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# NEW: Target Encoded Features (Risk Scores)
# We treat Surgery Name and Anesthesia as categories to find their average risk
target_cat_features = ['text_surgery', 'Anesthesia_Method', 'Patient_Source']
target_transformer = Pipeline(steps=[
    # TargetEncoder uses the 'y' labels to calculate average risk per category
    # smooth='auto' prevents overfitting on rare surgeries
    ('encoder', TargetEncoder(smooth='auto', target_type='continuous')),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

surgery_features = 'text_surgery'
surgery_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))),
    ('svd', TruncatedSVD(n_components=50, random_state=42))
])

rest_features = 'text_rest'
rest_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))),
    ('svd', TruncatedSVD(n_components=50, random_state=42))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('target', target_transformer, target_cat_features), # <--- NEW
        ('surg_text', surgery_transformer, surgery_features),
        ('rest_text', rest_transformer, rest_features)
    ])

# ==========================================
# 5. Models
# ==========================================

lr_model = LogisticRegression(max_iter=5000, C=1.0, solver='lbfgs', class_weight='balanced')
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=30, class_weight='balanced', random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='cosine')
svm_model = LinearSVC(C=0.5, class_weight='balanced', dual=False, random_state=42, max_iter=5000)

gb_model = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.03,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    early_stopping=True
)

rf_model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)

lgbm_model = None
if LIGHTGBM_AVAILABLE:
    # OPTIMIZED: Increased learning_rate to 0.05 (faster convergence)
    # Decreased n_estimators to 300 (faster training)
    # n_jobs=-1 allows it to use ALL cores when running standalone
    lgbm_model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)

models = {
    "Method #1 (Logistic Regression)": lr_model,
    "Method #2 (Decision Tree)": dt_model,
    "Method #3 (KNN - SVD Enhanced)": knn_model,
    "Method #4 (SVM - SVD Enhanced)": svm_model,
    "Method #5 (HistGradientBoosting)": gb_model
}
if lgbm_model:
    models["Method #6 (LightGBM)"] = lgbm_model

# Diversity Stack
estimators_list = [
    ('knn', knn_model),
    ('svm', svm_model),
    ('rf', rf_model)
]
if lgbm_model:
    estimators_list.append(('lgbm', lgbm_model))
else:
    estimators_list.append(('hgb', gb_model))

stacking_model = StackingClassifier(
    estimators=estimators_list,
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=3,       # OPTIMIZED: Reduce internal CV to 3 (Speed up Stacking)
    n_jobs=1    # OPTIMIZED: Process folds sequentially to avoid CPU contention with LightGBM
)
models["Method #7 (Stacking Ensemble)"] = stacking_model


# ==========================================
# 6. Training & Execution
# ==========================================

print(f"STARTING EXPERIMENTS (CV Folds: {CV_FOLDS})")
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
    
    # A. Local Validation
    clf.fit(X_train_split, y_train_split)
    val_preds = clf.predict(X_val_split)
    val_f1 = f1_score(y_val_split, val_preds, average='macro')
    
    print(f"   [Local Validation] Macro F1 Score ({int(VALIDATION_SPLIT_SIZE*100)}%): {val_f1:.4f}")
    
    # B. Cross Validation (Uses CV_FOLDS variable)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scoring = {
        'f1_macro': 'f1_macro', 'prec_macro': 'precision_macro', 'rec_macro': 'recall_macro',
        'f1_micro': 'f1_micro', 'prec_micro': 'precision_micro', 'rec_micro': 'recall_micro'
    }
    
    try:
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    except:
         scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=1)

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

    # C. Submission File
    print(f"   Retraining on FULL dataset...")
    clf.fit(X, y) 
    predictions = clf.predict(X_test)
    
    short_name = name.split('(')[1].split(')')[0].replace(" ", "")
    filename = f"kaggle_submission_{short_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    if len(submission_ids) != len(predictions):
        submission_ids = range(0, len(predictions))

    submission_df = pd.DataFrame({'Id': submission_ids, 'ASA_Rating': predictions})
    submission_df.to_csv(filename, index=False)
    print(f"   Saved: {filename}")

print("\n" + "="*60)
print(f"ALL DONE! Output recorded in 'output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt'.")