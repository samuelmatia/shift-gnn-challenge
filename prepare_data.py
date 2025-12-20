"""
Script to prepare the Super User temporal network data for the role transition prediction challenge.
Converts temporal graph data to parquet format and creates temporal snapshots with role assignments.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import os
import hashlib

def load_temporal_graph(filepath):
    """Load temporal graph from text file."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep=' ', header=None, names=['src', 'dst', 'timestamp'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('timestamp')
    print(f"Loaded {len(df)} edges from {df['datetime'].min()} to {df['datetime'].max()}")
    return df

def compute_user_features_per_period(edges_df, start_time, end_time):
    """
    Compute user features for a given time period.
    Returns a dict mapping user_id to features.
    """
    period_edges = edges_df[(edges_df['datetime'] >= start_time) & (edges_df['datetime'] < end_time)]
    
    user_features = defaultdict(lambda: {
        'out_degree': 0,
        'in_degree': 0,
        'num_answers': 0,
        'num_questions': 0,
        'num_comments': 0,
        'unique_recipients': set(),
        'unique_sources': set(),
        'first_interaction': None,
        'last_interaction': None,
        'total_interactions': 0
    })
    
    for _, row in period_edges.iterrows():
        src, dst, ts = row['src'], row['dst'], row['datetime']
        
        # Outgoing interactions
        user_features[src]['out_degree'] += 1
        user_features[src]['unique_recipients'].add(dst)
        user_features[src]['total_interactions'] += 1
        if user_features[src]['first_interaction'] is None:
            user_features[src]['first_interaction'] = ts
        user_features[src]['last_interaction'] = ts
        
        # Incoming interactions
        user_features[dst]['in_degree'] += 1
        user_features[dst]['unique_sources'].add(src)
        user_features[dst]['total_interactions'] += 1
        if user_features[dst]['first_interaction'] is None:
            user_features[dst]['first_interaction'] = ts
        user_features[dst]['last_interaction'] = ts
    
    # Convert to regular dict and compute final features
    result = {}
    for user_id, features in user_features.items():
        result[user_id] = {
            'out_degree': features['out_degree'],
            'in_degree': features['in_degree'],
            'num_unique_recipients': len(features['unique_recipients']),
            'num_unique_sources': len(features['unique_sources']),
            'total_interactions': features['total_interactions'],
            'first_interaction': features['first_interaction'],
            'last_interaction': features['last_interaction'],
            'activity_span_days': (features['last_interaction'] - features['first_interaction']).days + 1 if features['first_interaction'] else 0,
            'avg_interactions_per_day': features['total_interactions'] / max(1, (features['last_interaction'] - features['first_interaction']).days + 1) if features['first_interaction'] else 0
        }
    
    return result

def assign_role(user_features, thresholds):
    """
    Assign role to a user based on their features.
    Roles: 0=Inactive, 1=Novice, 2=Contributor, 3=Expert, 4=Moderator
    """
    total_interactions = user_features.get('total_interactions', 0)
    out_degree = user_features.get('out_degree', 0)
    in_degree = user_features.get('in_degree', 0)
    unique_recipients = user_features.get('num_unique_recipients', 0)
    
    # Inactive: no interactions (shouldn't happen in period_features, but handle it)
    if total_interactions == 0:
        return 0  # Inactive
    
    # Moderator: very high activity, helps many users (check first to avoid being classified as Expert)
    if total_interactions >= thresholds['moderator_min'] and unique_recipients >= thresholds['moderator_recipients']:
        return 4  # Moderator
    
    # Expert: high activity, many answers (high out_degree)
    if total_interactions >= thresholds['expert_min'] and out_degree >= thresholds['expert_out_degree']:
        return 3  # Expert
    
    # Novice: mostly asks questions (high in_degree relative to out_degree)
    # More lenient condition: in_degree > out_degree * 1.5 (instead of 2)
    if in_degree > out_degree * 1.5 and total_interactions < thresholds['contributor_min']:
        return 1  # Novice
    
    # Contributor: balanced interactions or default for medium activity
    if total_interactions >= thresholds['contributor_min']:
        return 2  # Contributor
    
    # Low activity but has some interactions - classify as Novice
    return 1  # Novice

def compute_thresholds_for_period(period_features):
    """
    Compute role assignment thresholds based on features from a specific period.
    This ensures thresholds are adapted to the snapshot duration (e.g., 3 months).
    """
    if not period_features:
        return {
            'novice_min': 1,
            'contributor_min': 3,
            'expert_min': 10,
            'expert_out_degree': 5,
            'moderator_min': 25,
            'moderator_recipients': 15
        }
    
    all_interactions = [f['total_interactions'] for f in period_features.values()]
    all_out_degrees = [f['out_degree'] for f in period_features.values()]
    all_unique_recipients = [f['num_unique_recipients'] for f in period_features.values()]
    
    # Use more lenient percentiles for 3-month windows
    # Lower thresholds to ensure we get all 5 roles
    thresholds = {
        'novice_min': 1,
        'contributor_min': max(2, int(np.percentile(all_interactions, 40))),  # Lower percentile
        'expert_min': max(5, int(np.percentile(all_interactions, 75))),  # Lower percentile
        'expert_out_degree': max(3, int(np.percentile(all_out_degrees, 70))),  # Lower percentile
        'moderator_min': max(15, int(np.percentile(all_interactions, 90))),  # Lower percentile
        'moderator_recipients': max(10, int(np.percentile(all_unique_recipients, 85)))  # Lower percentile
    }
    
    return thresholds

def create_temporal_snapshots(edges_df, snapshot_months=3):
    """
    Create temporal snapshots of the graph.
    Each snapshot covers snapshot_months months.
    Returns a list of DataFrames, one per snapshot.
    Includes inactive users (role 0) who were active before but not in current snapshot.
    """
    start_time = edges_df['datetime'].min()
    end_time = edges_df['datetime'].max()
    
    snapshots = []
    current_time = start_time
    previously_active_users = set()  # Track users who were active in previous snapshots
    
    snapshot_id = 0
    while current_time < end_time:
        snapshot_end = current_time + pd.DateOffset(months=snapshot_months)
        
        # Get all users active in this period
        period_features = compute_user_features_per_period(edges_df, current_time, snapshot_end)
        currently_active_users = set(period_features.keys())
        
        # Compute thresholds specific to this snapshot period
        thresholds = compute_thresholds_for_period(period_features)
        
        snapshot_data = []
        
        # Process active users
        for user_id, features in period_features.items():
            role = assign_role(features, thresholds)
            snapshot_data.append({
                'user_id': user_id,
                'snapshot_id': snapshot_id,
                'snapshot_start': current_time,
                'snapshot_end': snapshot_end,
                'role': role,
                **features
            })
        
        # Add inactive users (were active before but not now)
        inactive_users = set()
        if snapshot_id > 0:  # Skip first snapshot
            inactive_users = previously_active_users - currently_active_users
            for user_id in inactive_users:
                # Create minimal features for inactive user
                snapshot_data.append({
                    'user_id': user_id,
                    'snapshot_id': snapshot_id,
                    'snapshot_start': current_time,
                    'snapshot_end': snapshot_end,
                    'role': 0,  # Inactive
                    'out_degree': 0,
                    'in_degree': 0,
                    'num_unique_recipients': 0,
                    'num_unique_sources': 0,
                    'total_interactions': 0,
                    'first_interaction': None,
                    'last_interaction': None,
                    'activity_span_days': 0,
                    'avg_interactions_per_day': 0.0
                })
        
        # Update previously active users for next iteration
        previously_active_users = previously_active_users | currently_active_users
        
        if snapshot_data:
            snapshot_df = pd.DataFrame(snapshot_data)
            snapshots.append(snapshot_df)
            if snapshot_id == 0:  # Print thresholds only for first snapshot
                print(f"Role assignment thresholds (example for snapshot 0): {thresholds}")
            print(f"Snapshot {snapshot_id}: {current_time.date()} to {snapshot_end.date()} - {len(snapshot_df)} users ({len(currently_active_users)} active, {len(inactive_users)} inactive), roles: {sorted(snapshot_df['role'].unique())}")
        
        current_time = snapshot_end
        snapshot_id += 1
    
    return pd.concat(snapshots, ignore_index=True), thresholds

def create_role_transitions(snapshots_df):
    """
    Create role transition sequences for each user.
    For each user, create transitions: (role_t, role_t+1, role_t+2, ...)
    """
    transitions = []
    
    # Group by user and sort by snapshot_id
    for user_id, user_snapshots in snapshots_df.groupby('user_id'):
        user_snapshots = user_snapshots.sort_values('snapshot_id')
        
        if len(user_snapshots) < 2:
            continue  # Need at least 2 snapshots for a transition
        
        for i in range(len(user_snapshots) - 1):
            current = user_snapshots.iloc[i]
            next_snapshot = user_snapshots.iloc[i + 1]
            
            transition_label = f"{current['role']}->{next_snapshot['role']}"
            
            transitions.append({
                'user_id': user_id,
                'snapshot_id': current['snapshot_id'],
                'current_role': current['role'],
                'next_role': next_snapshot['role'],
                'transition_label': transition_label,
                'snapshot_start': current['snapshot_start'],
                'snapshot_end': current['snapshot_end'],
                'next_snapshot_start': next_snapshot['snapshot_start'],
                'next_snapshot_end': next_snapshot['snapshot_end'],
                # Include features from current snapshot
                'out_degree': current['out_degree'],
                'in_degree': current['in_degree'],
                'num_unique_recipients': current['num_unique_recipients'],
                'num_unique_sources': current['num_unique_sources'],
                'total_interactions': current['total_interactions'],
                'activity_span_days': current['activity_span_days'],
                'avg_interactions_per_day': current['avg_interactions_per_day']
            })
    
    return pd.DataFrame(transitions)

def anonymize_ids(df, secret_seed='CHALLENGE_SECRET_SEED'):
    """
    Anonymize user_id and snapshot_id using a secret seed.
    This prevents participants from re-running prepare_data.py to get labels.
    The mapping is deterministic but not obvious.
    """
    # Create deterministic but non-obvious mappings
    seed_int = int(hashlib.md5(secret_seed.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed_int)
    
    # Get unique IDs
    unique_user_ids = sorted(df['user_id'].unique())
    unique_snapshot_ids = sorted(df['snapshot_id'].unique())
    
    # Create random but deterministic mappings
    user_id_mapping = {uid: i for i, uid in enumerate(np.random.permutation(unique_user_ids))}
    snapshot_id_mapping = {sid: i for i, sid in enumerate(np.random.permutation(unique_snapshot_ids))}
    
    # Apply mappings
    df_anon = df.copy()
    df_anon['user_id'] = df_anon['user_id'].map(user_id_mapping)
    df_anon['snapshot_id'] = df_anon['snapshot_id'].map(snapshot_id_mapping)
    
    return df_anon, user_id_mapping, snapshot_id_mapping

def shuffle_test_set(test_df, secret_seed='CHALLENGE_TEST_SHUFFLE'):
    """
    Shuffle the test set rows using a different secret seed.
    This adds an extra layer of protection: even if someone re-runs the script,
    the order of test samples will be different, making it harder to match.
    """
    seed_int = int(hashlib.md5(secret_seed.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed_int)
    
    test_shuffled = test_df.copy()
    shuffled_indices = np.random.permutation(len(test_shuffled))
    test_shuffled = test_shuffled.iloc[shuffled_indices].reset_index(drop=True)
    
    return test_shuffled

def split_temporal_data(transitions_df, train_end='2014-12-31', 
                       anonymize_seed='CHALLENGE_SECRET_SEED',
                       shuffle_test_seed='CHALLENGE_TEST_SHUFFLE'):
    """
    Split data temporally:
    - Train: transitions before train_end
    - Test: transitions after train_end
    
    Then anonymize IDs and shuffle test set to prevent participants from 
    re-running the script to get labels.
    """
    train_end = pd.to_datetime(train_end)
    
    # First, do temporal split
    train = transitions_df[transitions_df['snapshot_start'] < train_end].copy()
    test = transitions_df[transitions_df['snapshot_start'] >= train_end].copy()
    
    print(f"\nTemporal split (before anonymization):")
    print(f"Train: {len(train)} transitions ({train['snapshot_start'].min().date()} to {train['snapshot_start'].max().date()})")
    print(f"Test: {len(test)} transitions ({test['snapshot_start'].min().date() if len(test) > 0 else 'N/A'} to {test['snapshot_start'].max().date() if len(test) > 0 else 'N/A'})")
    
    # Anonymize IDs to prevent reverse engineering
    # Combine train and test for consistent mapping, then split again
    all_data = pd.concat([train, test], ignore_index=True)
    all_data_anon, user_mapping, snapshot_mapping = anonymize_ids(all_data, anonymize_seed)
    
    # Split again after anonymization
    train_anon = all_data_anon[all_data_anon['snapshot_start'] < train_end].copy()
    test_anon = all_data_anon[all_data_anon['snapshot_start'] >= train_end].copy()
    
    # Shuffle test set with a different seed for extra protection
    test_anon_shuffled = shuffle_test_set(test_anon, shuffle_test_seed)
    
    print(f"\nAfter anonymization and shuffling:")
    print(f"Train: {len(train_anon)} transitions")
    print(f"Test: {len(test_anon_shuffled)} transitions (shuffled)")
    
    return train_anon, test_anon_shuffled, user_mapping, snapshot_mapping

def main():
    # Create output directory
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the main graph
    edges_df = load_temporal_graph('data/sx-superuser.txt')
    
    # Create temporal snapshots (3-month windows)
    print("\nCreating temporal snapshots...")
    snapshots_df, thresholds = create_temporal_snapshots(edges_df, snapshot_months=3)
    
    # Create role transitions
    print("\nCreating role transitions...")
    transitions_df = create_role_transitions(snapshots_df)
    
    # Print transition statistics
    print("\nTransition statistics:")
    print(transitions_df['transition_label'].value_counts().head(20))
    
    # Split temporally (only train and test, no validation)
    # Use secret seeds to anonymize IDs and shuffle test - KEEP THESE SECRET!
    # IMPORTANT: Change these seeds for production and DO NOT share them!
    
    ANONYMIZE_SEED = 'CHALLENGE_SECRET_SEED_2025'  # For ID anonymization
    SHUFFLE_TEST_SEED = 'CHALLENGE_TEST_SHUFFLE_2025'  # For test set shuffling
    
    train, test, user_mapping, snapshot_mapping = split_temporal_data(
        transitions_df, 
        train_end='2014-12-31',
        anonymize_seed=ANONYMIZE_SEED,
        shuffle_test_seed=SHUFFLE_TEST_SEED
    )
    
    # Save train set (public - for participants)
    train.to_parquet(f'{output_dir}/train.parquet', index=False)
    print(f"\nSaved train set to {output_dir}/train.parquet")
    
    # Save test features (public - for participants, WITHOUT labels)
    test_features = test.drop(columns=['next_role', 'transition_label', 'next_snapshot_start', 'next_snapshot_end']).copy()
    test_features.to_parquet(f'{output_dir}/test_features.parquet', index=False)
    print(f"Saved test features (without labels) to {output_dir}/test_features.parquet")
    
    # Save test set with labels (PRIVATE - only for organizers/scoring)
    # Store in a separate directory that should NOT be shared with participants
    private_dir = 'data/private'
    os.makedirs(private_dir, exist_ok=True)
    test.to_parquet(f'{private_dir}/test.parquet', index=False)
    
    # Save mappings for reference (also private)
    import pickle
    with open(f'{private_dir}/id_mappings.pkl', 'wb') as f:
        pickle.dump({'user_mapping': user_mapping, 'snapshot_mapping': snapshot_mapping}, f)
    
    print(f"Saved test set (with labels) to {private_dir}/test.parquet [PRIVATE - DO NOT SHARE]")
    print(f"Saved ID mappings to {private_dir}/id_mappings.pkl [PRIVATE - DO NOT SHARE]")
    
    print("\n" + "="*60)
    print("IMPORTANT: Keep the following files PRIVATE (do not share with participants):")
    print(f"  - {private_dir}/test.parquet")
    print(f"  - {private_dir}/id_mappings.pkl")
    print("="*60)
    
    print("\nData preparation complete!")
    print(f"Train: {len(train)} samples")
    print(f"Test: {len(test)} samples")
    print(f"  - Public: test_features.parquet (without labels)")
    print(f"  - Private: test.parquet (with labels, for scoring only)")

if __name__ == '__main__':
    main()

