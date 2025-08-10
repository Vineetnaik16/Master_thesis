# regime_evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Load data with regimes ---
df = pd.read_csv('spy_processed_data.csv', parse_dates=['Date'], index_col='Date')

# These should already exist from previous scripts
assert 'Clustering_regime_label' in df.columns, "Missing clustering label"
assert 'rule_20_regime' in df.columns, "Missing 20% rule label"

# --- Encode regimes consistently ---
regime_order = ['Bear', 'Bull']
le = LabelEncoder()
le.classes_ = np.array(regime_order)

true_labels = le.transform(df['rule_20_regime'])
pred_labels = le.transform(df['Clustering_regime_label'])

# --- Accuracy ---
accuracy = accuracy_score(true_labels, pred_labels)
print(f"\n✅ Clustering vs. 20% Rule Accuracy: {accuracy:.2%}")

# --- Confusion Matrix ---
labels = regime_order
cm = pd.DataFrame(
    confusion_matrix(df['rule_20_regime'], df['Clustering_regime_label'], labels=labels),
    index=[f"Actual_{l}" for l in labels],
    columns=[f"Pred_{l}" for l in labels]
)
print("\n📊 Confusion Matrix:\n", cm)

# --- Precision per predicted regime ---
precisions = {}
for label in labels:
    col = f"Pred_{label}"
    true_pos = cm.at[f"Actual_{label}", col]
    pred_total = cm[col].sum()
    precision = true_pos / pred_total if pred_total else 0
    precisions[label] = precision

print("\n🎯 Precision by predicted regime:")
for label, val in precisions.items():
    print(f"Precision for predicted '{label}': {val:.2%}")

# --- Strategy vs Buy & Hold Returns ---
df['strategy_return'] = df['log_return'] * (df['Clustering_regime_label'] == 'Bull').astype(int)
df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod()
df['cumulative_bah'] = (1 + df['log_return']).cumprod()

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['cumulative_bah'], label='Buy & Hold', color='gray')
plt.plot(df.index, df['cumulative_strategy'], label='Regime Strategy', color='blue')
plt.legend()
plt.title('📈 Strategy vs. Buy & Hold (Regime Based)')
plt.grid(True)
plt.show()

# --- Predict Next Day's Regime (optional) ---
df['target_regime'] = df['Clustering_regime_label'].shift(-1)
df = df.dropna(subset=['target_regime'])

features = ['log_return', 'volatility', 'drawdown', 'ma_distance']
X = df[features]
y = le.transform(df['target_regime'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n🔍 Predicting Next Day's Regime:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
