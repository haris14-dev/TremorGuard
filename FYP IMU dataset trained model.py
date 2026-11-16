import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("Dataset.csv")
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Label distribution:\n{df['Result'].value_counts()}")

sensor_cols = ["aX", "aY", "aZ", "gX", "gY", "gZ"]
X = df[sensor_cols].values
y = df["Result"].values

def extract_features(segment):
    features = []
    for i in range(segment.shape[1]):
        col = segment[:, i]
        features.extend([
            np.mean(col),
            np.std(col),
            np.min(col),
            np.max(col),
            np.sqrt(np.mean(col**2))
        ])
    sma = np.mean(np.sum(np.abs(segment), axis=1))
    features.append(sma)
    return np.array(features)

def segment_signal(data, labels, window=100, overlap=50):
    segments = []
    seg_labels = []
    start = 0
    while start + window <= len(data):
        segments.append(data[start:start+window])
        seg_labels.append(int(np.round(np.mean(labels[start:start+window]))))
        start += overlap
    return segments, np.array(seg_labels)

print("Extracting features from signal segments...")
segments, segment_labels = segment_signal(X, y, window=100, overlap=50)
print(f"Created {len(segments)} segments")

X_features = np.array([extract_features(seg) for seg in segments])
print(f"Features shape: {X_features.shape}")


print("Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_features, segment_labels, test_size=0.2, random_state=42, stratify=segment_labels
)

print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining 3 models...\n")
models = {}
results = {}
cv_results = {}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)


rf_cv_acc = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='accuracy')
rf_cv_f1 = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='f1')

models['rf'] = rf_model
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred),
    'recall': recall_score(y_test, rf_pred),
    'f1': f1_score(y_test, rf_pred)
}
cv_results['Random Forest'] = {
    'cv_accuracy': f"{rf_cv_acc.mean():.4f} ± {rf_cv_acc.std():.4f}",
    'cv_f1': f"{rf_cv_f1.mean():.4f} ± {rf_cv_f1.std():.4f}"
}
print(f"       Test Accuracy: {results['Random Forest']['accuracy']:.4f}")
print(f"       CV Accuracy: {rf_cv_acc.mean():.4f} ± {rf_cv_acc.std():.4f}")
print(f"       CV F1-Score: {rf_cv_f1.mean():.4f} ± {rf_cv_f1.std():.4f}")

print("Training SVM...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)


svm_cv_acc = cross_val_score(svm_model, X_train, y_train, cv=skf, scoring='accuracy')
svm_cv_f1 = cross_val_score(svm_model, X_train, y_train, cv=skf, scoring='f1')

models['svm'] = svm_model
results['SVM'] = {
    'accuracy': accuracy_score(y_test, svm_pred),
    'precision': precision_score(y_test, svm_pred),
    'recall': recall_score(y_test, svm_pred),
    'f1': f1_score(y_test, svm_pred)
}
cv_results['SVM'] = {
    'cv_accuracy': f"{svm_cv_acc.mean():.4f} ± {svm_cv_acc.std():.4f}",
    'cv_f1': f"{svm_cv_f1.mean():.4f} ± {svm_cv_f1.std():.4f}"
}
print(f"       Test Accuracy: {results['SVM']['accuracy']:.4f}")
print(f"       CV Accuracy: {svm_cv_acc.mean():.4f} ± {svm_cv_acc.std():.4f}")
print(f"       CV F1-Score: {svm_cv_f1.mean():.4f} ± {svm_cv_f1.std():.4f}")

print("Training KNN...")
knn_model = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)


knn_cv_acc = cross_val_score(knn_model, X_train, y_train, cv=skf, scoring='accuracy')
knn_cv_f1 = cross_val_score(knn_model, X_train, y_train, cv=skf, scoring='f1')

models['knn'] = knn_model
results['KNN'] = {
    'accuracy': accuracy_score(y_test, knn_pred),
    'precision': precision_score(y_test, knn_pred),
    'recall': recall_score(y_test, knn_pred),
    'f1': f1_score(y_test, knn_pred)
}
cv_results['KNN'] = {
    'cv_accuracy': f"{knn_cv_acc.mean():.4f} ± {knn_cv_acc.std():.4f}",
    'cv_f1': f"{knn_cv_f1.mean():.4f} ± {knn_cv_f1.std():.4f}"
}
print(f"       Test Accuracy: {results['KNN']['accuracy']:.4f}")
print(f"       CV Accuracy: {knn_cv_acc.mean():.4f} ± {knn_cv_acc.std():.4f}")
print(f"       CV F1-Score: {knn_cv_f1.mean():.4f} ± {knn_cv_f1.std():.4f}")


print("MODEL PERFORMANCE SUMMARY")

results_df = pd.DataFrame(results).T
print(results_df.to_string())

print("\n--- 5-Fold Cross-Validation Results ---")
cv_results_df = pd.DataFrame(cv_results).T
print(cv_results_df.to_string())

print("\nBest Model (Test Accuracy):", results_df['accuracy'].idxmax(),
      f"({results_df['accuracy'].max():.4f})")


cv_means = {k: float(v['cv_accuracy'].split(' ')[0]) for k, v in cv_results.items()}
best_cv_model = max(cv_means, key=cv_means.get)
print("Best Model (CV Accuracy):", best_cv_model,
      f"({cv_means[best_cv_model]:.4f})")

print("\nSaving models...")
joblib.dump(models['rf'], "rf_model.pkl")
joblib.dump(models['svm'], "svm_model.pkl")
joblib.dump(models['knn'], "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("All models and scaler saved.")

print("TRAINING COMPLETE")
