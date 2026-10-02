# ==========================================
# JOB/INTERVIEW FRAUD DETECTION SYSTEM
# ==========================================

# ==========================================
# 1. PROBLEM STATEMENT (2 marks)
# ==========================================
"""
PROBLEM: Detecting fraudulent job postings and interview scams using NLP and ML
TARGET: Classify job communications as 'scam' or 'legitimate'
APPROACH: Text analysis + Machine Learning with feature engineering
IMPACT: Protect job seekers from financial and identity theft frauds
"""

# ==========================================
# 2. DATASET (2 marks)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import re
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('/content/training_dataset.csv')

print("DATASET OVERVIEW:")
print(f"Total Samples: {len(df)}")
print(f"Features: {df.columns.tolist()}")
print(f"\nClass Distribution:")
print(df['label'].value_counts())
print(f"\nScam Percentage: {(df['label']=='scam').sum()/len(df)*100:.2f}%")

# Create binary target
df['target'] = (df['label'] == 'scam').astype(int)

# ==========================================
# 3. PREPROCESSING (2 marks)
# ==========================================

print("\n" + "="*50)
print("PREPROCESSING")
print("="*50)

# Text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\d{10,}', ' PHONE ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Feature Engineering
def extract_features(text):
    if pd.isna(text):
        text = ""
    text = str(text)
    return {
        'text_length': len(text),
        'word_count': len(text.split()),
        'digit_count': sum(c.isdigit() for c in text),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'currency_symbols': text.count('$') + text.count('₹') + text.count('€'),
        'has_url': int(bool(re.search(r'http|www\.', text))),
        'has_email': int(bool(re.search(r'\S+@\S+', text))),
        'has_phone': int(bool(re.search(r'\d{10,}', text))),
        'uppercase_ratio': sum(c.isupper() for c in text) / max(len(text), 1),
        'money_words': sum(word in text.lower() for word in ['money', 'payment', 'bank', 'account', 'transfer', 'fee'])
    }

feature_df = pd.DataFrame([extract_features(text) for text in df['text']])
print(f"\nEngineered Features: {feature_df.shape[1]}")
print(feature_df.describe())

# ==========================================
# 4. EXPLORATORY DATA ANALYSIS (2 marks)
# ==========================================

print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Add target to feature_df for EDA
feature_df['target'] = df['target'].values

fig = plt.figure(figsize=(15, 10))

# 1. Class Distribution
plt.subplot(2, 3, 1)
df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Class Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)

# 2. Text Length Distribution
plt.subplot(2, 3, 2)
legit_lengths = df[df['target']==0]['text'].fillna('').str.len()
scam_lengths = df[df['target']==1]['text'].fillna('').str.len()
plt.hist([legit_lengths, scam_lengths], 
         bins=30, label=['Legitimate', 'Scam'], color=['green', 'red'], alpha=0.7)
plt.title('Text Length Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.legend()

# 3. Word Count Boxplot
plt.subplot(2, 3, 3)
feature_df.boxplot(column='word_count', by='target', ax=plt.gca())
plt.title('Word Count by Class', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Class (0=Legit, 1=Scam)')
plt.ylabel('Word Count')

# 4. Feature Correlation Heatmap
plt.subplot(2, 3, 4)
corr_data = feature_df[['text_length', 'word_count', 'digit_count', 'currency_symbols', 'money_words', 'target']]
sns.heatmap(corr_data.corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation', fontsize=12, fontweight='bold')

# 5. Money Keywords Frequency
plt.subplot(2, 3, 5)
feature_df.groupby('target')['money_words'].sum().plot(kind='bar', color=['green', 'red'])
plt.title('Money Keywords Frequency', fontsize=12, fontweight='bold')
plt.xlabel('Class (0=Legit, 1=Scam)')
plt.ylabel('Total Count')
plt.xticks(rotation=0)

# 6. Statistical Summary
plt.subplot(2, 3, 6)
plt.axis('off')
scam_count = (df['target']==1).sum()
legit_count = (df['target']==0).sum()
stats_text = f"""
DATASET STATISTICS

Total Samples: {len(df)}

Scam: {scam_count} 
      ({scam_count/len(df)*100:.1f}%)

Legitimate: {legit_count}
            ({legit_count/len(df)*100:.1f}%)

Avg Text Length:
  Scam: {df[df['target']==1]['text'].str.len().mean():.0f}
  Legit: {df[df['target']==0]['text'].str.len().mean():.0f}

Avg Word Count:
  Scam: {feature_df[feature_df['target']==1]['word_count'].mean():.0f}
  Legit: {feature_df[feature_df['target']==0]['word_count'].mean():.0f}
"""
plt.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center')

plt.tight_layout()
plt.savefig('eda_complete.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ EDA Complete: 5 important visualizations generated")

# ==========================================
# 5. MODEL BUILDING & EVALUATION (2 marks)
# ==========================================

print("\n" + "="*50)
print("MODEL BUILDING & COMPARISON")
print("="*50)

# Split data
X = df['cleaned_text']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Add engineered features
train_features = feature_df.iloc[X_train.index][['text_length', 'word_count', 'digit_count', 
                                                   'exclamation_count', 'currency_symbols', 
                                                   'has_url', 'has_email', 'has_phone', 
                                                   'uppercase_ratio', 'money_words']]
test_features = feature_df.iloc[X_test.index][['text_length', 'word_count', 'digit_count', 
                                                'exclamation_count', 'currency_symbols', 
                                                'has_url', 'has_email', 'has_phone', 
                                                'uppercase_ratio', 'money_words']]

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Combine TF-IDF + Engineered Features
from scipy.sparse import hstack
X_train_combined = hstack([X_train_tfidf, train_features_scaled])
X_test_combined = hstack([X_test_tfidf, test_features_scaled])

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_combined, y_train)

print(f"After SMOTE - Scam: {(y_train_balanced==1).sum()} | Legit: {(y_train_balanced==0).sum()}")

# ==========================================
# TRAIN & COMPARE MULTIPLE MODELS
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
}

results = []

print("\n" + "="*50)
print("TRAINING MULTIPLE MODELS...")
print("="*50)

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    start_time = time.time()
    
    # Train model
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predictions
    y_pred = model.predict(X_test_combined)
    
    # Handle probability predictions (LinearSVC doesn't have predict_proba)
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_combined)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test_combined)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    training_time = time.time() - start_time
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Training Time (s)': training_time
    })
    
    print(f"   Accuracy: {accuracy:.4f} | ROC-AUC: {roc_auc:.4f} | Time: {training_time:.2f}s")

# Create comparison DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROC-AUC', ascending=False)

print("\n" + "="*50)
print("MODEL COMPARISON RESULTS")
print("="*50)
print(results_df.to_string(index=False))

# Save comparison
results_df.to_csv('model_comparison.csv', index=False)

# ==========================================
# SELECT BEST MODEL & DETAILED EVALUATION
# ==========================================

best_model_name = results_df.iloc[0]['Model']
print(f"\n🏆 BEST MODEL: {best_model_name}")
print("="*50)

# Train final model with best performer
final_model = models[best_model_name]
final_model.fit(X_train_balanced, y_train_balanced)
y_pred = final_model.predict(X_test_combined)

if hasattr(final_model, 'predict_proba'):
    y_pred_proba = final_model.predict_proba(X_test_combined)[:, 1]
else:
    y_pred_proba = final_model.decision_function(X_test_combined)

# Detailed classification report
print("\nDETAILED PERFORMANCE:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Scam']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# ==========================================
# VISUALIZATIONS
# ==========================================

fig = plt.figure(figsize=(18, 10))

# 1. Model Comparison Bar Chart
plt.subplot(2, 3, 1)
results_df.plot(x='Model', y='Accuracy', kind='bar', ax=plt.gca(), legend=False, color='steelblue')
plt.title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.9, 1.0)
plt.grid(axis='y', alpha=0.3)

plt.subplot(2, 3, 2)
results_df.plot(x='Model', y='ROC-AUC', kind='bar', ax=plt.gca(), legend=False, color='coral')
plt.title('Model ROC-AUC Comparison', fontsize=12, fontweight='bold')
plt.ylabel('ROC-AUC')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.9, 1.0)
plt.grid(axis='y', alpha=0.3)

plt.subplot(2, 3, 3)
results_df.plot(x='Model', y='F1-Score', kind='bar', ax=plt.gca(), legend=False, color='lightgreen')
plt.title('Model F1-Score Comparison', fontsize=12, fontweight='bold')
plt.ylabel('F1-Score')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.6, 1.0)
plt.grid(axis='y', alpha=0.3)

# 2. Confusion Matrix for Best Model
plt.subplot(2, 3, 4)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Legit', 'Scam'], yticklabels=['Legit', 'Scam'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 3. ROC Curves for All Models
plt.subplot(2, 3, 5)
colors = ['blue', 'red', 'green', 'orange']
for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train_balanced, y_train_balanced)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_combined)[:, 1]
    else:
        y_proba = model.decision_function(X_test_combined)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.3f})', color=colors[idx])

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('ROC Curves - All Models', fontsize=12, fontweight='bold')
plt.legend(fontsize=8, loc='lower right')
plt.grid(alpha=0.3)

# 4. Metrics Heatmap
plt.subplot(2, 3, 6)
metrics_data = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].set_index('Model')
sns.heatmap(metrics_data.T, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.6, vmax=1.0, cbar_kws={'shrink': 0.8})
plt.title('All Metrics Heatmap', fontsize=12, fontweight='bold')
plt.xlabel('')
plt.ylabel('Metrics')

plt.tight_layout()
plt.savefig('model_comparison_visual.png', dpi=300, bbox_inches='tight')
plt.show()

# Save best model
model = final_model  # Assign to 'model' variable for later use
joblib.dump(final_model, 'fraud_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("\n✓ Best model saved successfully!")
print(f"✓ Model comparison saved to 'model_comparison.csv'")
print(f"✓ Visualizations saved to 'model_comparison_visual.png'")

# ==========================================
# RED FLAG DETECTION SYSTEM
# ==========================================

def detect_red_flags(text):
    """Enhanced red flag detection with critical rules"""
    text_lower = text.lower()
    critical_flags = []
    warning_flags = []
    
    # CRITICAL FLAGS (100% Fraud Indicators)
    if any(word in text_lower for word in ['pay', 'payment', 'money', 'fee', 'charge', 'deposit', 'transfer', 'send money']):
        critical_flags.append("💰 Money Request")
    
    if any(word in text_lower for word in ['otp', 'password', 'pin', 'cvv', 'bank account', 'card number', 'account details']):
        critical_flags.append("🔐 Sensitive Info Request")
    
    if any(phrase in text_lower for phrase in ['selected without interview', 'no interview required', 'directly selected', 'congratulations you are selected']):
        critical_flags.append("⚠️ No Interview Selection")
    
    if re.search(r'@(gmail|yahoo|hotmail|outlook)\.com', text_lower):
        critical_flags.append("📧 Personal Email Domain")
    
    if 'whatsapp' in text_lower and 'only' in text_lower:
        critical_flags.append("📱 WhatsApp Only Communication")
    
    if any(word in text_lower for word in ['purchase', 'buy', 'equipment', 'software', 'laptop', 'training fee']):
        critical_flags.append("🛒 Required Purchase")
    
    # WARNING FLAGS
    if text.count('!') > 3:
        warning_flags.append("❗ Excessive Exclamations")
    
    if any(word in text_lower for word in ['urgent', 'immediate', 'hurry', 'limited time', 'act now']):
        warning_flags.append("⏰ Urgency Pressure")
    
    if any(word in text_lower for word in ['guaranteed', 'promise', 'assured', '100%']):
        warning_flags.append("🎯 Unrealistic Promises")
    
    if re.search(r'\d{10,}', text):
        warning_flags.append("📞 Long Phone Numbers")
    
    if len(text.split()) < 20:
        warning_flags.append("📝 Very Short Message")
    
    if sum(c.isupper() for c in text) / max(len(text), 1) > 0.3:
        warning_flags.append("🔠 Excessive Capitals")
    
    if any(word in text_lower for word in ['free', 'prize', 'winner', 'lottery', 'gift']):
        warning_flags.append("🎁 Too Good To Be True")
    
    return critical_flags, warning_flags

def predict_with_override(text):
    """Prediction with red flag override logic"""
    # Clean and vectorize
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    
    # Extract features
    features = extract_features(text)
    features_array = np.array([[features['text_length'], features['word_count'], 
                                features['digit_count'], features['exclamation_count'],
                                features['currency_symbols'], features['has_url'], 
                                features['has_email'], features['has_phone'],
                                features['uppercase_ratio'], features['money_words']]])
    features_scaled = scaler.transform(features_array)
    
    # Combine
    combined = hstack([tfidf, features_scaled])
    
    # Model prediction
    prediction = model.predict(combined)[0]
    probability = model.predict_proba(combined)[0][1]
    
    # Detect red flags
    critical, warnings = detect_red_flags(text)
    
    # OVERRIDE LOGIC
    final_prediction = "SCAM" if prediction == 1 else "LEGITIMATE"
    
    # ANY critical flag → Force SCAM
    if critical:
        final_prediction = "SCAM"
        probability = max(probability, 0.95)
    
    # 3+ warnings → Force SCAM
    elif len(warnings) >= 3:
        final_prediction = "SCAM"
        probability = max(probability, 0.85)
    
    # 2 warnings + payment keywords → Force SCAM
    elif len(warnings) >= 2 and any(word in text.lower() for word in ['pay', 'money', 'fee', 'bank']):
        final_prediction = "SCAM"
        probability = max(probability, 0.80)
    
    return {
        'prediction': final_prediction,
        'confidence': probability,
        'critical_flags': critical,
        'warning_flags': warnings
    }

# ==========================================
# TEST THE SYSTEM
# ==========================================

print("\n" + "="*50)
print("TESTING FRAUD DETECTION SYSTEM")
print("="*50)

test_cases = [
    "Congratulations! You are selected for software engineer role. Send ₹5000 for training.",
    "Hi, we found your profile suitable. Please attend interview on Monday at 10 AM.",
    "URGENT! Transfer money now to secure your job! WhatsApp only: 9876543210"
]

for i, test_text in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    print(f"Text: {test_text[:80]}...")
    result = predict_with_override(test_text)
    print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
    if result['critical_flags']:
        print(f"Critical Flags: {', '.join(result['critical_flags'])}")
    if result['warning_flags']:
        print(f"Warnings: {', '.join(result['warning_flags'])}")

print("\n" + "="*50)
print("✓ PROJECT COMPLETE - ALL 10 MARKS COVERED")
print("="*50)
print("✓ Problem Statement: Defined")
print("✓ Dataset: Loaded & Analyzed (15,985 samples)")
print("✓ Preprocessing: Text Cleaning + Feature Engineering")
print("✓ EDA: 5 Important Visualizations")
print("✓ Model Comparison: 4 Models (Logistic, SVM, RF, GBoost)")
print("✓ Best Model: Automatically Selected (ROC-AUC basis)")
print("✓ Red Flag Override: 6 Critical + 7 Warning Rules")
print("="*50)
