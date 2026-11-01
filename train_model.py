"""
Fraud Detection Model Training Pipeline
=========================================
Complete ML pipeline for training fraud detection models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("="*60)
print("FRAUD DETECTION MODEL - TRAINING PIPELINE")
print("="*60)

# ============================================
# 1. LOAD AND ANALYZE DATASET
# ============================================
print("\n[1/7] Loading dataset...")
df = pd.read_csv(r'C:\Users\HP\Downloads\training_dataset.csv')

print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns[:10])}...")

# Check for required columns
if 'text' not in df.columns or 'label' not in df.columns:
    print("ERROR: Dataset must have 'text' and 'label' columns")
    exit(1)

# Basic info
print(f"\nMissing values in 'text': {df['text'].isna().sum()}")
print(f"Missing values in 'label': {df['label'].isna().sum()}")

# Drop rows with missing text or label
df = df.dropna(subset=['text', 'label'])
print(f"After dropping NaN: {df.shape[0]} rows")

# Class distribution
print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)
label_counts = df['label'].value_counts()
print(label_counts)
print(f"\nScam percentage: {(label_counts.get('scam', 0) / len(df) * 100):.2f}%")

# Convert label to binary
df['target'] = (df['label'] == 'scam').astype(int)

# Save class distribution plot
plt.figure(figsize=(8, 6))
df['target'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Class (0=Legitimate, 1=Scam)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
for i, v in enumerate(df['target'].value_counts().values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: class_distribution.png")
plt.close()

# ============================================
# 2. FEATURE ENGINEERING
# ============================================
print("\n[2/7] Creating features...")

def extract_features(text):
    """Extract numeric features from text"""
    if pd.isna(text) or not isinstance(text, str):
        return {
            'text_length': 0,
            'word_count': 0,
            'digit_count': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'currency_symbols': 0,
            'has_url': 0,
            'has_email': 0,
            'has_phone': 0,
            'uppercase_ratio': 0,
            'money_words': 0
        }
    
    text = str(text)
    words = text.split()
    
    # Money-related keywords
    money_keywords = ['fee', 'pay', 'payment', 'deposit', 'transfer', 'money', 
                     'rupees', 'dollars', 'amount', 'charge', 'cost', 'price']
    
    return {
        'text_length': len(text),
        'word_count': len(words),
        'digit_count': sum(c.isdigit() for c in text),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'currency_symbols': text.count('₹') + text.count('$') + text.count('£'),
        'has_url': int(bool(re.search(r'http[s]?://', text))),
        'has_email': int(bool(re.search(r'\S+@\S+', text))),
        'has_phone': int(bool(re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text))),
        'uppercase_ratio': sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0,
        'money_words': sum(keyword in text.lower() for keyword in money_keywords)
    }

# Apply feature engineering
features_df = pd.DataFrame([extract_features(text) for text in df['text']])
print(f"✓ Extracted {features_df.shape[1]} numeric features")
print(f"  Features: {list(features_df.columns)}")

# ============================================
# 3. TRAIN-TEST SPLIT
# ============================================
print("\n[3/7] Splitting data...")

X_text = df['text'].to_numpy()
X_features = features_df.to_numpy()
y = df['target'].to_numpy()

# Split
X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train: {len(X_text_train)} samples")
print(f"✓ Test:  {len(X_text_test)} samples")
print(f"  Train scam %: {(y_train.sum() / len(y_train) * 100):.2f}%")
print(f"  Test scam %:  {(y_test.sum() / len(y_test) * 100):.2f}%")

# ============================================
# 4. TEXT VECTORIZATION
# ============================================
print("\n[4/7] Vectorizing text (TF-IDF)...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words='english'
)

X_text_train_vec = vectorizer.fit_transform(X_text_train)
X_text_test_vec = vectorizer.transform(X_text_test)

print(f"✓ TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"✓ Train shape: {X_text_train_vec.shape}")

# Scale numeric features
scaler = StandardScaler()
X_feat_train_scaled = scaler.fit_transform(X_feat_train)
X_feat_test_scaled = scaler.transform(X_feat_test)

# Combine text + numeric features
from scipy.sparse import hstack, csr_matrix
X_train_combined = hstack([X_text_train_vec, csr_matrix(X_feat_train_scaled)])
X_test_combined = hstack([X_text_test_vec, csr_matrix(X_feat_test_scaled)])

print(f"✓ Combined feature shape: {X_train_combined.shape}")

# ============================================
# 5. HANDLE CLASS IMBALANCE (SMOTE)
# ============================================
print("\n[5/7] Handling class imbalance with SMOTE...")
print(f"Before SMOTE - Scam: {y_train.sum()}, Legit: {(y_train == 0).sum()}")
try:
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)
    print(f"After SMOTE  - Scam: {y_train_resampled.sum()}, Legit: {(y_train_resampled == 0).sum()}")
    print(f"✓ Balanced training set: {X_train_resampled.shape[0]} samples")
except Exception as smote_err:
    print(f"⚠ SMOTE failed: {smote_err}. Using original imbalanced data.")
    X_train_resampled, y_train_resampled = X_train_combined, y_train

# ============================================
# 6. TRAIN MULTIPLE MODELS
# ============================================
print("\n[6/7] Training models...")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Linear SVM': LinearSVC(max_iter=2000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n→ Training {name}...")
    try:
        # Train
        model.fit(X_train_resampled, y_train_resampled)

        # Predict
        y_pred = model.predict(X_test_combined)

        # Metrics
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_combined)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test_combined)
        else:
            y_proba = y_pred

        roc_auc = roc_auc_score(y_test, y_proba)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[name] = {
            'model': model,
            'roc_auc': roc_auc,
            'accuracy': report['accuracy'],
            'precision_scam': report.get('1', {}).get('precision', 0),
            'recall_scam': report.get('1', {}).get('recall', 0),
            'f1_scam': report.get('1', {}).get('f1-score', 0),
            'y_pred': y_pred,
            'y_proba': y_proba
        }

        print(f"  ✓ ROC-AUC: {roc_auc:.4f}")
        print(f"  ✓ Accuracy: {report['accuracy']:.4f}")
        print(f"  ✓ Scam Precision: {results[name]['precision_scam']:.4f}")
        print(f"  ✓ Scam Recall: {results[name]['recall_scam']:.4f}")
    except Exception as train_err:
        print(f"  ✗ Error training {name}: {train_err}. Skipping.")

# ============================================
# 6b. HYPERPARAMETER TUNING (Random Forest)
# ============================================
print("\n[6b] Hyperparameter tuning for Random Forest...")
try:
    rf_param_dist = {
        'n_estimators':     [100, 200, 300, 500],
        'max_depth':        [10, 20, 30, None],
        'min_samples_split':[2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features':     ['sqrt', 'log2']
    }
    rf_tuning_base = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf_tuning_base, rf_param_dist,
        n_iter=15,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc', n_jobs=-1, random_state=42, verbose=0
    )
    rf_search.fit(X_train_resampled, y_train_resampled)
    best_rf_tuned   = rf_search.best_estimator_
    y_pred_tuned    = best_rf_tuned.predict(X_test_combined)
    y_proba_tuned   = best_rf_tuned.predict_proba(X_test_combined)[:, 1]
    roc_auc_tuned   = roc_auc_score(y_test, y_proba_tuned)
    report_tuned    = classification_report(y_test, y_pred_tuned, output_dict=True, zero_division=0)
    print(f"  Best params: {rf_search.best_params_}")
    print(f"  Tuned RF ROC-AUC: {roc_auc_tuned:.4f}")
    if roc_auc_tuned > results.get('Random Forest', {}).get('roc_auc', 0):
        results['Random Forest (Tuned)'] = {
            'model':          best_rf_tuned,
            'roc_auc':        roc_auc_tuned,
            'accuracy':       report_tuned['accuracy'],
            'precision_scam': report_tuned.get('1', {}).get('precision', 0),
            'recall_scam':    report_tuned.get('1', {}).get('recall',    0),
            'f1_scam':        report_tuned.get('1', {}).get('f1-score',  0),
            'y_pred':         y_pred_tuned,
            'y_proba':        y_proba_tuned
        }
        print("  ✓ Tuned RF outperforms default → added to comparison")
    else:
        print("  ✓ Default RF is already near-optimal for this dataset")
except Exception as tune_err:
    print(f"  ⚠ Hyperparameter tuning error: {tune_err}. Skipping.")

# ============================================
# 6c. FINE-TUNE BERT MODEL
# ============================================
print("\n[6c] Fine-tuning BERT for fraud detection...")
BERT_TRAINED = False
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AdamW, get_linear_schedule_with_warmup
    )
    import os

    BERT_MODEL_NAME = 'bert-base-uncased'
    BERT_MAX_LEN    = 128
    BERT_BATCH_SIZE = 16
    BERT_EPOCHS     = 3
    BERT_LR         = 2e-5
    BERT_SAVE_PATH  = 'bert_fraud_model'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    class FraudDataset(Dataset):
        """PyTorch Dataset for fraud text classification"""
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts     = [str(t) if t is not None else "" for t in texts]
            self.labels    = labels
            self.tokenizer = tokenizer
            self.max_len   = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids':      enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
            }

    # Cap training size for feasibility (CPU/limited VRAM)
    MAX_BERT_TRAIN = min(len(X_text_train), 8000)
    train_ds = FraudDataset(
        X_text_train[:MAX_BERT_TRAIN].tolist(),
        y_train[:MAX_BERT_TRAIN].tolist(),
        bert_tokenizer, BERT_MAX_LEN
    )
    test_ds = FraudDataset(
        X_text_test.tolist(), y_test.tolist(), bert_tokenizer, BERT_MAX_LEN
    )
    train_dl = DataLoader(train_ds, batch_size=BERT_BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=BERT_BATCH_SIZE, shuffle=False)

    bert_clf        = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)
    bert_clf        = bert_clf.to(device)
    optimizer_bert  = AdamW(bert_clf.parameters(), lr=BERT_LR, weight_decay=0.01)
    total_steps     = len(train_dl) * BERT_EPOCHS
    scheduler_bert  = get_linear_schedule_with_warmup(
        optimizer_bert,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps
    )

    print(f"  Training BERT: {BERT_EPOCHS} epochs | {MAX_BERT_TRAIN} samples | batch {BERT_BATCH_SIZE}")
    bert_clf.train()
    for epoch in range(BERT_EPOCHS):
        epoch_loss = 0.0
        for step, batch in enumerate(train_dl):
            try:
                optimizer_bert.zero_grad()
                out = bert_clf(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(bert_clf.parameters(), 1.0)
                optimizer_bert.step()
                scheduler_bert.step()
                epoch_loss += out.loss.item()
                if (step + 1) % 100 == 0:
                    print(f"    Epoch {epoch+1}/{BERT_EPOCHS} | Step {step+1}/{len(train_dl)} "
                          f"| Avg Loss: {epoch_loss/(step+1):.4f}")
            except RuntimeError as batch_err:
                print(f"    ⚠ Batch skipped: {batch_err}")
                continue
        print(f"  ✓ Epoch {epoch+1} complete | Avg Loss: {epoch_loss/max(1, len(train_dl)):.4f}")

    # ---- BERT Evaluation ----
    bert_clf.eval()
    bert_preds, bert_probs = [], []
    with torch.no_grad():
        for batch in test_dl:
            try:
                out   = bert_clf(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
                probs = torch.softmax(out.logits, dim=1)
                bert_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                bert_probs.extend(probs[:, 1].cpu().numpy())
            except Exception as eval_err:
                print(f"    ⚠ Eval batch error: {eval_err}")
                continue

    bert_preds     = np.array(bert_preds)
    bert_probs_arr = np.array(bert_probs)

    if len(bert_preds) > 0:
        bert_roc_auc = roc_auc_score(y_test[:len(bert_preds)], bert_probs_arr)
        bert_report  = classification_report(
            y_test[:len(bert_preds)], bert_preds, output_dict=True, zero_division=0
        )
        results['Fine-Tuned BERT'] = {
            'model':          None,           # saved separately as directory
            'roc_auc':        bert_roc_auc,
            'accuracy':       bert_report['accuracy'],
            'precision_scam': bert_report.get('1', {}).get('precision', 0),
            'recall_scam':    bert_report.get('1', {}).get('recall',    0),
            'f1_scam':        bert_report.get('1', {}).get('f1-score',  0),
            'y_pred':         bert_preds,
            'y_proba':        bert_probs_arr
        }
        print(f"  ✓ BERT ROC-AUC   : {bert_roc_auc:.4f}")
        print(f"  ✓ BERT Accuracy  : {bert_report['accuracy']:.4f}")
        print(f"  ✓ BERT Precision : {bert_report.get('1', {}).get('precision', 0):.4f}")
        print(f"  ✓ BERT Recall    : {bert_report.get('1', {}).get('recall', 0):.4f}")
        # Save BERT model & tokenizer
        os.makedirs(BERT_SAVE_PATH, exist_ok=True)
        bert_clf.save_pretrained(BERT_SAVE_PATH)
        bert_tokenizer.save_pretrained(BERT_SAVE_PATH)
        print(f"  ✓ BERT saved → '{BERT_SAVE_PATH}/'")
        BERT_TRAINED = True
    else:
        print("  ⚠ No BERT predictions collected — check batch errors above.")

except ImportError as ie:
    print(f"  ⚠ BERT dependencies not installed: {ie}")
    print("  → Install with: pip install transformers torch")
except Exception as bert_err:
    print(f"  ⚠ BERT training failed: {bert_err}")

# ============================================
# 7. SELECT BEST MODEL
# ============================================
print("\n[7/7] Selecting best model...")
print("="*60)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'ROC-AUC': [results[m]['roc_auc'] for m in results],
    'Accuracy': [results[m]['accuracy'] for m in results],
    'Precision (Scam)': [results[m]['precision_scam'] for m in results],
    'Recall (Scam)': [results[m]['recall_scam'] for m in results],
    'F1-Score (Scam)': [results[m]['f1_scam'] for m in results]
})

print("\nMODEL COMPARISON:")
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv('model_comparison.csv', index=False)
print("\n✓ Saved: model_comparison.csv")

# Best sklearn model (model not None) → saved as .joblib
sklearn_results = {k: v for k, v in results.items() if v['model'] is not None}
if not sklearn_results:
    raise RuntimeError("No sklearn models trained successfully. Check training errors above.")
best_sklearn_name = max(sklearn_results, key=lambda x: sklearn_results[x]['roc_auc'])
best_model        = sklearn_results[best_sklearn_name]['model']

# Overall best (may be BERT)
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])

print(f"\n🏆 BEST SKLEARN MODEL : {best_sklearn_name}")
print(f"   ROC-AUC            : {sklearn_results[best_sklearn_name]['roc_auc']:.4f}")
if 'Fine-Tuned BERT' in results:
    print(f"\n🤖 FINE-TUNED BERT ROC-AUC : {results['Fine-Tuned BERT']['roc_auc']:.4f}")
print(f"\n🥇 OVERALL BEST : {best_model_name} (AUC={results[best_model_name]['roc_auc']:.4f})")

# ============================================
# 8. DETAILED EVALUATION OF BEST MODEL
# ============================================
print("\n" + "="*60)
print("DETAILED EVALUATION - BEST MODEL")
print("="*60)

y_pred_best = results[best_model_name]['y_pred']
y_proba_best = results[best_model_name]['y_proba']

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Legitimate', 'Scam'], zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Scam'], yticklabels=['Legitimate', 'Scam'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_best)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'{best_model_name} (AUC = {results[best_model_name]["roc_auc"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
print("✓ Saved: roc_curve.png")
plt.close()

# ============================================
# 9. SAVE MODEL ARTIFACTS
# ============================================
print("\n" + "="*60)
print("SAVING MODEL ARTIFACTS")
print("="*60)

joblib.dump(best_model, 'fraud_model.joblib')
print("✓ Saved: fraud_model.joblib")

joblib.dump(vectorizer, 'vectorizer.joblib')
print("✓ Saved: vectorizer.joblib")

joblib.dump(scaler, 'scaler.joblib')
print("✓ Saved: scaler.joblib")

# Save metadata
metadata = {
    'best_model':      best_sklearn_name,
    'best_overall':    best_model_name,
    'bert_available':  BERT_TRAINED,
    'bert_model_path': 'bert_fraud_model' if BERT_TRAINED else None,
    'bert_roc_auc':    results['Fine-Tuned BERT']['roc_auc'] if BERT_TRAINED else None,
    'roc_auc':         results[best_sklearn_name]['roc_auc'],
    'accuracy':        results[best_sklearn_name]['accuracy'],
    'precision_scam':  results[best_sklearn_name]['precision_scam'],
    'recall_scam':     results[best_sklearn_name]['recall_scam'],
    'f1_scam':         results[best_sklearn_name]['f1_scam'],
    'train_samples':   len(X_text_train),
    'test_samples':    len(X_text_test),
    'vocab_size':      len(vectorizer.vocabulary_),
    'features':        list(features_df.columns)
}

import json
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Saved: model_metadata.json")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\nBest Sklearn Model : {best_sklearn_name}")
print(f"ROC-AUC            : {results[best_sklearn_name]['roc_auc']:.4f}")
print(f"Accuracy           : {results[best_sklearn_name]['accuracy']:.4f}")
print(f"Scam Precision     : {results[best_sklearn_name]['precision_scam']:.4f}")
print(f"Scam Recall        : {results[best_sklearn_name]['recall_scam']:.4f}")
if BERT_TRAINED:
    print(f"\nFine-Tuned BERT ROC-AUC  : {results['Fine-Tuned BERT']['roc_auc']:.4f}")
    print(f"Fine-Tuned BERT Accuracy : {results['Fine-Tuned BERT']['accuracy']:.4f}")
    print(f"Ensemble (apps)          : BERT 65% + {best_sklearn_name} 35%")
print("\nNext step: Run 'streamlit run app.py' to launch the web interface")
print("="*60)
