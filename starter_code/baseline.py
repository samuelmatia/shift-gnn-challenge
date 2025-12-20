"""
Simple baseline model for Role Transition Prediction challenge.
Uses Random Forest on graph-based features extracted from temporal snapshots.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import sys
import os

# Add parent directory to path to import scoring function
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    """Load training and validation data.
    Uses temporal split: last 20% of training period as validation.
    This respects the temporal nature of the data (no data leakage).
    """
    train_full = pd.read_parquet('./data/processed/train.parquet')
    
    # Temporal split: use last 20% of training period as validation
    # This respects the temporal nature of the data (no data leakage)
    train_full['snapshot_start'] = pd.to_datetime(train_full['snapshot_start'])
    train_full = train_full.sort_values('snapshot_start')
    
    # Find the cutoff date for 80/20 temporal split
    cutoff_idx = int(len(train_full) * 0.8)
    cutoff_date = train_full.iloc[cutoff_idx]['snapshot_start']
    
    train = train_full[train_full['snapshot_start'] < cutoff_date].copy()
    val = train_full[train_full['snapshot_start'] >= cutoff_date].copy()
    
    return train, val

def prepare_features(df):
    """Extract features for model training."""
    feature_cols = [
        'current_role',
        'out_degree',
        'in_degree',
        'num_unique_recipients',
        'num_unique_sources',
        'total_interactions',
        'activity_span_days',
        'avg_interactions_per_day'
    ]
    
    X = df[feature_cols].fillna(0)
    return X

def train_baseline():
    """Train baseline Random Forest model."""
    print("Loading data...")
    train, val = load_data()
    
    print(f"Train set: {len(train)} samples")
    print(f"Val set: {len(val)} samples")
    
    # Prepare features
    X_train = prepare_features(train)
    y_train = train['next_role']
    
    X_val = prepare_features(val)
    y_val = val['next_role']
    
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_pred = clf.predict(X_val)
    
    # Compute Macro-F1 score
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    print(f"\nMacro-F1 Score: {macro_f1:.4f}")
    
    # Get unique classes present in validation set
    # y_val is a pandas Series, y_pred is a numpy array
    unique_classes = sorted(set(y_val.unique()) | set(np.unique(y_pred)))
    role_names = ['Inactive', 'Novice', 'Contributor', 'Expert', 'Moderator']
    
    # Print classification report with only present classes
    print("\nClassification Report:")
    print(f"Classes present in validation: {unique_classes}")
    print(classification_report(
        y_val, y_pred, 
        labels=unique_classes,
        target_names=[role_names[i] for i in unique_classes],
        zero_division=0
    ))
    
    # Compute F1 for rare transitions
    print("\nTransition-level analysis:")
    val_with_pred = val.copy()
    val_with_pred['predicted_role'] = y_pred
    val_with_pred['true_transition'] = val_with_pred['current_role'].astype(str) + '->' + val_with_pred['next_role'].astype(str)
    val_with_pred['pred_transition'] = val_with_pred['current_role'].astype(str) + '->' + val_with_pred['predicted_role'].astype(str)
    
    # Focus on rare transitions (e.g., transitions that occur < 5% of the time)
    transition_counts = val_with_pred['true_transition'].value_counts()
    rare_threshold = len(val_with_pred) * 0.05
    rare_transitions = transition_counts[transition_counts < rare_threshold].index.tolist()
    
    print(f"\nRare transitions (< 5%): {len(rare_transitions)}")
    if rare_transitions:
        rare_mask = val_with_pred['true_transition'].isin(rare_transitions)
        rare_f1 = f1_score(
            val_with_pred.loc[rare_mask, 'next_role'],
            val_with_pred.loc[rare_mask, 'predicted_role'],
            average='macro'
        )
        print(f"Macro-F1 on rare transitions: {rare_f1:.4f}")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    test_features = pd.read_parquet('./data/processed/test_features.parquet')
    X_test = prepare_features(test_features)
    test_preds = clf.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'user_id': test_features['user_id'],
        'snapshot_id': test_features['snapshot_id'],
        'predicted_role': test_preds
    })
    
    os.makedirs('./submissions', exist_ok=True)
    submission.to_csv('./submissions/sample_submission_1.csv', index=False)
    print("Saved predictions to ./submissions/sample_submission_1.csv")
    
    return clf

if __name__ == '__main__':
    train_baseline()

