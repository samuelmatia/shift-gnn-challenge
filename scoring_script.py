"""
Scoring script for Role Transition Prediction challenge.
Primary metric: Weighted Macro-F1 Score (inverse frequency weighting).
Also computes Overall Macro-F1 and Rare Transitions Macro-F1.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
import sys

def compute_rare_transition_f1(y_true, y_pred, current_roles, rare_threshold_percentile=5):
    """
    Compute Macro-F1 score focusing on rare transitions.
    
    Args:
        y_true: True next roles
        y_pred: Predicted next roles
        current_roles: Current roles (for transition labels)
        rare_threshold_percentile: Percentile threshold for rare transitions
    
    Returns:
        Dictionary with various F1 scores
    """
    # Create transition labels
    true_transitions = [f"{c}->{n}" for c, n in zip(current_roles, y_true)]
    pred_transitions = [f"{c}->{n}" for c, n in zip(current_roles, y_pred)]
    
    # Count transition frequencies
    transition_counts = pd.Series(true_transitions).value_counts()
    total_transitions = len(true_transitions)
    rare_threshold = np.percentile(transition_counts.values, rare_threshold_percentile)
    
    # Identify rare transitions
    rare_transitions = transition_counts[transition_counts <= rare_threshold].index.tolist()
    
    # Overall Macro-F1
    overall_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Macro-F1 on rare transitions only
    rare_mask = pd.Series(true_transitions).isin(rare_transitions)
    if rare_mask.sum() > 0:
        rare_f1 = f1_score(
            np.array(y_true)[rare_mask.values],
            np.array(y_pred)[rare_mask.values],
            average='macro'
        )
    else:
        rare_f1 = 0.0
    
    # Weighted F1: give more weight to rare transitions
    # Weight = 1 / frequency (normalized)
    transition_weights = {}
    for trans in set(true_transitions):
        freq = transition_counts.get(trans, 1)
        transition_weights[trans] = 1.0 / max(freq, 1)  # Inverse frequency weighting
    
    weights = [transition_weights.get(trans, 1.0) for trans in true_transitions]
    weighted_f1 = f1_score(y_true, y_pred, average='macro', sample_weight=weights)
    
    return {
        'overall_macro_f1': overall_f1,
        'rare_transitions_f1': rare_f1,
        'weighted_f1': weighted_f1,
        'num_rare_transitions': len(rare_transitions),
        'rare_transition_rate': rare_mask.sum() / len(true_transitions),
        'rare_transitions': rare_transitions[:10]  # Show first 10
    }

def score_submission(submission_file, ground_truth_file='data/private/test.parquet'):
    """
    Score a submission file.
    
    Args:
        submission_file: Path to submission CSV file
        ground_truth_file: Path to ground truth parquet file
    """
    print(f"Loading submission: {submission_file}")
    submission = pd.read_csv(submission_file)
    
    print(f"Loading ground truth: {ground_truth_file}")
    ground_truth = pd.read_parquet(ground_truth_file)
    
    # Merge to align predictions with ground truth
    merged = ground_truth.merge(
        submission,
        on=['user_id', 'snapshot_id'],
        how='inner',
        suffixes=('_true', '_pred')
    )
    
    if len(merged) == 0:
        print("ERROR: No matching predictions found!")
        return
    
    print(f"\nScoring {len(merged)} predictions...")
    
    y_true = merged['next_role'].values
    y_pred = merged['predicted_role'].values
    current_roles = merged['current_role'].values
    
    # Compute scores
    scores = compute_rare_transition_f1(y_true, y_pred, current_roles)
    
    print("\n" + "="*60)
    print("SCORING RESULTS")
    print("="*60)
    print(f"*** PRIMARY METRIC ***")
    print(f"Weighted Macro-F1 Score: {scores['weighted_f1']:.6f}")
    print(f"\nAdditional Metrics:")
    print(f"Overall Macro-F1 Score: {scores['overall_macro_f1']:.6f}")
    print(f"Rare Transitions Macro-F1 Score: {scores['rare_transitions_f1']:.6f}")
    print(f"\nRare Transitions Info:")
    print(f"  Number of rare transitions: {scores['num_rare_transitions']}")
    print(f"  Rare transition rate: {scores['rare_transition_rate']:.2%}")
    print(f"  Sample rare transitions: {scores['rare_transitions']}")
    print("="*60)
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Inactive', 'Novice', 'Contributor', 'Expert', 'Moderator']
    ))
    
    # Transition-level analysis
    print("\nTransition-level Performance:")
    merged['true_transition'] = merged['current_role'].astype(str) + '->' + merged['next_role'].astype(str)
    merged['pred_transition'] = merged['current_role'].astype(str) + '->' + merged['predicted_role'].astype(str)
    
    transition_f1 = {}
    for trans in merged['true_transition'].unique():
        mask = merged['true_transition'] == trans
        if mask.sum() > 0:
            trans_f1 = f1_score(
                merged.loc[mask, 'next_role'],
                merged.loc[mask, 'predicted_role'],
                average='macro'
            )
            transition_f1[trans] = {
                'f1': trans_f1,
                'count': mask.sum()
            }
    
    # Sort by frequency and show top transitions
    sorted_transitions = sorted(
        transition_f1.items(),
        key=lambda x: x[1]['count'],
        reverse=False  # Show rare ones first
    )
    
    print("\nTop 20 transitions (sorted by frequency, rare first):")
    print(f"{'Transition':<20} {'Count':<10} {'F1 Score':<10}")
    print("-" * 40)
    for trans, metrics in sorted_transitions[:20]:
        print(f"{trans:<20} {metrics['count']:<10} {metrics['f1']:.4f}")
    
    return scores

if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Score a submission file')
    parser.add_argument('submission_file', help='Path to submission CSV file')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--ground-truth', default='data/private/test.parquet', 
                       help='Path to ground truth file')
    
    args = parser.parse_args()
    
    scores = score_submission(args.submission_file, args.ground_truth)
    
    if args.json and scores:
        print("\n" + "="*60)
        print("JSON OUTPUT:")
        print("="*60)
        print(json.dumps(scores, indent=2))

