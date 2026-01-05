# src/train.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.features import featurize_dataset
import joblib

def build_feature_matrix():
    X_basic_df, X_window_mat, y, meta = featurize_dataset()
    # combine basic (tabular) + window (one-hot)
    X_combined = np.hstack([X_basic_df.values, X_window_mat])
    return X_combined, y, X_basic_df.columns, meta

def train_and_eval():
    X, y, basic_cols, meta = build_feature_matrix()
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight='balanced'
    )
    cv = StratifiedKFold(n_splits=min(5, len(y)))
    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    clf.fit(X, y)
    print(f"CV ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    # feature importance (only for basic features)
    importances = clf.feature_importances_[:len(basic_cols)]
    fi = pd.Series(importances, index=basic_cols).sort_values(ascending=False)
    print("Top feature importances:")
    print(fi.head(10))
    # save model
    
    joblib.dump(clf, "model_rf.joblib")
    return clf, fi

if __name__ == "__main__":
    train_and_eval()
