"""
Steel Defect Classification System
===================================
Dataset: UCI Steel Plates Faults Dataset (1941 samples, 27 features, 7 defect types)
Source: https://archive.ics.uci.edu/dataset/198/steel+plates+faults

Run this first to train the model and generate charts.
Then run: streamlit run app.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

os.makedirs('charts', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("📦 Loading Steel Plates Faults Dataset...")

# Try UCI API first; fall back to local CSV (same structure, same counts)
try:
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=198)
    X = dataset.data.features
    y_raw = dataset.data.targets
    defect_names = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    y_raw.columns = defect_names
    y = y_raw.idxmax(axis=1)
    print("✅ Loaded from UCI ML Repository")
except Exception:
    print("⚠️  UCI API unavailable — using local dataset (same structure & distribution)")
    df_local = pd.read_csv('steel_faults_data.csv')
    y = df_local['Defect_Type']
    X = df_local.drop(columns=['Defect_Type'])

defect_names = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"📊 Defect types: {y.unique().tolist()}")
print()

# ─────────────────────────────────────────────
# 2. EXPLORATORY ANALYSIS & CHARTS
# ─────────────────────────────────────────────

# SAIL-style color palette (steel/industrial theme)
COLORS = ['#1a3a5c', '#2e6da4', '#4a9dd4', '#e8a020', '#d45c1a', '#8b2020', '#2e8b57']
SAIL_BLUE = '#1a3a5c'
SAIL_ORANGE = '#e8a020'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})

# Chart 1: Defect Frequency
print("📈 Generating Chart 1: Defect Frequency...")
defect_counts = y.value_counts().reindex(defect_names).fillna(0)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(defect_counts.index, defect_counts.values, color=COLORS, edgecolor='white', linewidth=1.5)
ax.set_title('Steel Defect Frequency Distribution\n(UCI Steel Plates Dataset — 1941 Samples)', 
             fontsize=14, fontweight='bold', color=SAIL_BLUE, pad=15)
ax.set_xlabel('Defect Type', fontsize=11, color=SAIL_BLUE)
ax.set_ylabel('Number of Occurrences', fontsize=11, color=SAIL_BLUE)
ax.set_xticklabels(defect_counts.index, rotation=30, ha='right', fontsize=9)

for bar, val in zip(bars, defect_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            str(int(val)), ha='center', va='bottom', fontsize=9, fontweight='bold', color=SAIL_BLUE)

ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('charts/01_defect_frequency.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/01_defect_frequency.png")

# Chart 2: Feature Importance (preview with correlation)
print("📈 Generating Chart 2: Feature Correlations with defect types...")
df_full = X.copy()
df_full['Defect'] = y.values

le = LabelEncoder()
df_full['Defect_Code'] = le.fit_transform(df_full['Defect'])

top_features = ['Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity',
                'Luminosity_Index', 'SigmoidOfAreas', 'LogOfAreas', 'Edges_Index']

corr = df_full[top_features + ['Defect_Code']].corr()['Defect_Code'].drop('Defect_Code').sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = [SAIL_ORANGE if v > 0 else SAIL_BLUE for v in corr.values]
ax.barh(corr.index, corr.values, color=colors_bar, edgecolor='white')
ax.set_title('Feature Correlation with Defect Type', fontsize=13, fontweight='bold', color=SAIL_BLUE)
ax.set_xlabel('Pearson Correlation Coefficient', fontsize=10)
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('charts/02_feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/02_feature_correlation.png")

# ─────────────────────────────────────────────
# 3. TRAIN ML MODELS
# ─────────────────────────────────────────────
print("\n🤖 Training ML Models...")

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
    results[name] = {
        'model': model,
        'accuracy': acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred
    }
    print(f"   ✅ {name}: Test Acc = {acc:.4f} | CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Best model = Random Forest
best_model = results['Random Forest']['model']
best_pred = results['Random Forest']['y_pred']

# Save model
with open('models/steel_defect_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le_target, f)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("\n💾 Model saved to models/")

# ─────────────────────────────────────────────
# 4. MODEL EVALUATION CHARTS
# ─────────────────────────────────────────────

# Chart 3: Model Comparison
print("\n📈 Generating Chart 3: Model Accuracy Comparison...")
fig, ax = plt.subplots(figsize=(8, 4))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] * 100 for m in model_names]
cv_means = [results[m]['cv_mean'] * 100 for m in model_names]

x = np.arange(len(model_names))
w = 0.35
bars1 = ax.bar(x - w/2, accuracies, w, label='Test Accuracy', color=SAIL_BLUE, alpha=0.9)
bars2 = ax.bar(x + w/2, cv_means, w, label='CV Accuracy (5-fold)', color=SAIL_ORANGE, alpha=0.9)

for bar, val in zip(bars1, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=9)
ax.set_ylabel('Accuracy (%)', fontsize=10)
ax.set_title('ML Model Comparison — Steel Defect Classification', fontsize=12, fontweight='bold', color=SAIL_BLUE)
ax.legend()
ax.set_ylim(50, 105)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('charts/03_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/03_model_comparison.png")

# Chart 4: Confusion Matrix
print("📈 Generating Chart 4: Confusion Matrix...")
cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_, yticklabels=le_target.classes_,
            linewidths=0.5, ax=ax)
ax.set_title('Confusion Matrix — Random Forest Model\n(Steel Defect Classification)', 
             fontsize=13, fontweight='bold', color=SAIL_BLUE)
ax.set_xlabel('Predicted Defect', fontsize=10)
ax.set_ylabel('Actual Defect', fontsize=10)
plt.xticks(rotation=35, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('charts/04_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/04_confusion_matrix.png")

# Chart 5: Feature Importance (from Random Forest)
print("📈 Generating Chart 5: Feature Importance...")
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True)
top15 = importances.tail(15)

fig, ax = plt.subplots(figsize=(9, 6))
colors_fi = [SAIL_ORANGE if i >= 12 else SAIL_BLUE for i in range(len(top15))]
top15.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
ax.set_title('Top 15 Features Influencing Steel Defect Type\n(Random Forest Feature Importance)', 
             fontsize=12, fontweight='bold', color=SAIL_BLUE)
ax.set_xlabel('Importance Score', fontsize=10)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('charts/05_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/05_feature_importance.png")

# ─────────────────────────────────────────────
# 5. SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  STEEL DEFECT CLASSIFICATION — PROJECT SUMMARY")
print("="*55)
print(f"  Dataset       : UCI Steel Plates Faults")
print(f"  Total Samples : {X.shape[0]}")
print(f"  Features      : {X.shape[1]}")
print(f"  Defect Types  : 7")
print()
print(f"  Best Model    : Random Forest")
print(f"  Test Accuracy : {results['Random Forest']['accuracy']*100:.2f}%")
print(f"  CV Accuracy   : {results['Random Forest']['cv_mean']*100:.2f}%")
print()
print("  Charts saved in /charts/")
print("  Model saved in /models/")
print()
print("  ➜  Next: run  streamlit run app.py")
print("="*55)

print("\n📋 Classification Report (Random Forest):")
print(classification_report(y_test, best_pred, target_names=le_target.classes_))
