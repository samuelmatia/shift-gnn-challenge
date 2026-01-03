"""
Advanced baseline model for Role Transition Prediction challenge.
Uses GraphSAGE (PyTorch Geometric) + LSTM to capture both graph structure and temporal dynamics.
This should outperform the simple Random Forest baseline.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
import os
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class GraphSAGEModel(nn.Module):
    """GraphSAGE model using PyTorch Geometric."""
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2):
        super(GraphSAGEModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(SAGEConv(in_feats, hidden_feats))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_feats, hidden_feats))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(SAGEConv(hidden_feats, out_feats))
        else:
            self.layers.append(SAGEConv(in_feats, out_feats))
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, feat_dim]
            edge_index: Edge indices [2, num_edges]
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

class RoleTransitionPredictor(nn.Module):
    """Complete model: GraphSAGE + LSTM for temporal role transition prediction."""
    def __init__(self, node_feat_dim, hidden_dim=128, num_roles=5, num_layers=2):
        super(RoleTransitionPredictor, self).__init__()
        
        # GraphSAGE for learning node embeddings
        self.graphsage = GraphSAGEModel(node_feat_dim, hidden_dim, hidden_dim, num_layers)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(hidden_dim + node_feat_dim + 5, hidden_dim, batch_first=True, num_layers=1)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_roles)
        )
        
    def forward(self, x, edge_index, user_ids, current_roles):
        """
        Forward pass.
        
        Args:
            x: Node features tensor [num_nodes, feat_dim]
            edge_index: Edge indices [2, num_edges]
            user_ids: User IDs for the batch [batch_size]
            current_roles: Current roles [batch_size]
        """
        # Get graph embeddings for all nodes
        graph_embeddings = self.graphsage(x, edge_index)  # [num_nodes, hidden_dim]
        
        # Get embeddings for users in batch
        user_embeddings = graph_embeddings[user_ids]  # [batch_size, hidden_dim]
        
        # Combine with original features
        user_features = x[user_ids]  # [batch_size, feat_dim]
        
        # Clamp values to prevent extreme values
        user_embeddings = torch.clamp(user_embeddings, min=-10.0, max=10.0)
        user_features = torch.clamp(user_features, min=-10.0, max=10.0)
        
        combined = torch.cat([user_embeddings, user_features], dim=1)  # [batch_size, hidden_dim + feat_dim]
        
        # Add current role as one-hot encoding
        role_onehot = F.one_hot(current_roles.long(), num_classes=5).float()  # [batch_size, 5]
        combined = torch.cat([combined, role_onehot], dim=1)  # [batch_size, hidden_dim + feat_dim + 5]
        
        # Clamp combined features
        combined = torch.clamp(combined, min=-10.0, max=10.0)
        
        # LSTM processing (treat as sequence of length 1 for now)
        lstm_input = combined.unsqueeze(1)  # [batch_size, 1, hidden_dim + feat_dim + 5]
        lstm_out, _ = self.lstm(lstm_input)  # [batch_size, 1, hidden_dim]
        lstm_out = lstm_out.squeeze(1)  # [batch_size, hidden_dim]
        
        # Clamp LSTM output
        lstm_out = torch.clamp(lstm_out, min=-10.0, max=10.0)
        
        # Combine graph and LSTM outputs
        final_features = torch.cat([user_embeddings, lstm_out], dim=1)  # [batch_size, hidden_dim * 2]
        
        # Clamp final features
        final_features = torch.clamp(final_features, min=-10.0, max=10.0)
        
        # Classify
        logits = self.classifier(final_features)  # [batch_size, num_roles]
        
        return logits

class TransitionDataset(Dataset):
    """Dataset for role transitions."""
    def __init__(self, transitions_df, user_to_idx, node_features_dict):
        self.transitions = transitions_df.reset_index(drop=True)
        self.user_to_idx = user_to_idx
        self.node_features_dict = node_features_dict
        
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        row = self.transitions.iloc[idx]
        user_id = row['user_id']
        
        # Handle users not in mapping (use a default index if missing)
        if user_id not in self.user_to_idx:
            # Use a default index (0) for unknown users
            user_idx = 0
            # Use default features if user not in dict
            if user_id not in self.node_features_dict:
                # Return zero features
                features = np.zeros(8, dtype=np.float32)
            else:
                features = self.node_features_dict[user_id]
        else:
            user_idx = self.user_to_idx[user_id]
            features = self.node_features_dict.get(user_id, np.zeros(8, dtype=np.float32))
        
        current_role = row['current_role']
        # next_role may not exist in test_features.parquet
        next_role = row.get('next_role', -1)  # Use -1 as default if not present
        
        return {
            'user_idx': user_idx,
            'current_role': current_role,
            'next_role': next_role,
            'features': features
        }

def build_graph_from_transitions(transitions_df, max_edges_per_snapshot=1000):
    """
    Build a PyTorch Geometric graph from transitions data.
    Creates edges between users in the same snapshot, but limits to avoid memory issues.
    """
    # Get all unique users
    all_users = set(transitions_df['user_id'].unique())
    user_to_idx = {uid: idx for idx, uid in enumerate(sorted(all_users))}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    
    # Build edges from transitions (users who appear in same snapshot)
    edges = set()
    
    for snapshot_id in transitions_df['snapshot_id'].unique():
        snapshot_users = transitions_df[transitions_df['snapshot_id'] == snapshot_id]['user_id'].unique()
        snapshot_user_indices = [user_to_idx[uid] for uid in snapshot_users]
        
        # Limit number of edges per snapshot to avoid memory issues
        if len(snapshot_user_indices) > max_edges_per_snapshot:
            sampled_indices = np.random.choice(snapshot_user_indices, 
                                             size=max_edges_per_snapshot, 
                                             replace=False)
        else:
            sampled_indices = snapshot_user_indices
        
        # Create edges between sampled users
        for u1 in sampled_indices:
            others = [u for u in sampled_indices if u != u1]
            if len(others) > 0:
                k = min(5, len(others))
                neighbors = np.random.choice(others, size=k, replace=False)
                for u2 in neighbors:
                    edges.add((u1, u2))
                    edges.add((u2, u1))  # Undirected
    
    # Add self-loops for all nodes
    num_nodes = len(user_to_idx)
    for i in range(num_nodes):
        edges.add((i, i))
    
    if len(edges) == 0:
        # Fallback: create a simple graph with self-loops only
        edges = [(i, i) for i in range(num_nodes)]
    
    # Convert to edge_index format [2, num_edges]
    src, dst = zip(*edges)
    edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    
    return edge_index, user_to_idx, idx_to_user

def prepare_node_features(transitions_df, user_to_idx):
    """Prepare node features from transitions data."""
    # Aggregate features per user (use mean of features across all snapshots)
    user_features = defaultdict(lambda: {
        'out_degree': [],
        'in_degree': [],
        'num_unique_recipients': [],
        'num_unique_sources': [],
        'total_interactions': [],
        'activity_span_days': [],
        'avg_interactions_per_day': [],
        'current_role': []
    })
    
    for _, row in transitions_df.iterrows():
        uid = row['user_id']
        user_features[uid]['out_degree'].append(row['out_degree'])
        user_features[uid]['in_degree'].append(row['in_degree'])
        user_features[uid]['num_unique_recipients'].append(row['num_unique_recipients'])
        user_features[uid]['num_unique_sources'].append(row['num_unique_sources'])
        user_features[uid]['total_interactions'].append(row['total_interactions'])
        user_features[uid]['activity_span_days'].append(row['activity_span_days'])
        user_features[uid]['avg_interactions_per_day'].append(row['avg_interactions_per_day'])
        user_features[uid]['current_role'].append(row['current_role'])
    
    # Compute mean features
    node_features_dict = {}
    feature_matrix = []
    
    for uid in sorted(user_to_idx.keys()):
        feats = user_features[uid]
        mean_features = [
            np.mean(feats['out_degree']),
            np.mean(feats['in_degree']),
            np.mean(feats['num_unique_recipients']),
            np.mean(feats['num_unique_sources']),
            np.mean(feats['total_interactions']),
            np.mean(feats['activity_span_days']),
            np.mean(feats['avg_interactions_per_day']),
            np.mean(feats['current_role'])
        ]
        node_features_dict[uid] = np.array(mean_features, dtype=np.float32)
        feature_matrix.append(mean_features)
    
    feature_matrix = np.array(feature_matrix, dtype=np.float32)
    
    return node_features_dict, feature_matrix

def load_data():
    """Load training and validation data."""
    train_full = pd.read_parquet('./data/processed/train.parquet')
    
    # Temporal split
    train_full['snapshot_start'] = pd.to_datetime(train_full['snapshot_start'])
    train_full = train_full.sort_values('snapshot_start')
    
    cutoff_idx = int(len(train_full) * 0.8)
    cutoff_date = train_full.iloc[cutoff_idx]['snapshot_start']
    
    train = train_full[train_full['snapshot_start'] < cutoff_date].copy()
    val = train_full[train_full['snapshot_start'] >= cutoff_date].copy()
    
    return train, val

def train_model():
    """Train the advanced GNN model."""
    print("Loading data...")
    train, val = load_data()
    
    print(f"Train set: {len(train)} samples")
    print(f"Val set: {len(val)} samples")
    
    # Build graph from train, but include all users from train+val for mapping
    print("\nBuilding graph...")
    # Combine train and val to get all unique users
    all_transitions = pd.concat([train, val], ignore_index=True)
    edge_index, user_to_idx, idx_to_user = build_graph_from_transitions(train)
    
    # Add users from val that are not in train to the mapping
    train_users = set(train['user_id'].unique())
    val_users = set(val['user_id'].unique())
    new_users = val_users - train_users
    
    # Add new users to mapping (they won't be in the graph, but we need the mapping)
    max_idx = max(user_to_idx.values()) if user_to_idx else -1
    for uid in sorted(new_users):
        if uid not in user_to_idx:
            max_idx += 1
            user_to_idx[uid] = max_idx
    
    num_nodes = len(user_to_idx)
    print(f"Graph: {num_nodes} nodes ({len(train_users)} from train, {len(new_users)} new in val), {edge_index.shape[1]} edges")
    
    # Prepare node features (from train, but add defaults for new users)
    print("Preparing node features...")
    node_features_dict, feature_matrix = prepare_node_features(train, user_to_idx)
    
    # Add default features for new users in val
    if len(new_users) > 0:
        default_features = np.mean(feature_matrix, axis=0) if len(feature_matrix) > 0 else np.zeros(8, dtype=np.float32)
        for uid in new_users:
            if uid not in node_features_dict:
                node_features_dict[uid] = default_features
    
    feature_dim = feature_matrix.shape[1]
    
    # Check for NaN or inf in features
    print("Checking for NaN/inf in features...")
    nan_count = np.isnan(feature_matrix).sum()
    inf_count = np.isinf(feature_matrix).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"Warning: Found {nan_count} NaN and {inf_count} inf values. Replacing...")
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Normalize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # Check again after normalization
    nan_count = np.isnan(feature_matrix_scaled).sum()
    inf_count = np.isinf(feature_matrix_scaled).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"Warning: Found {nan_count} NaN and {inf_count} inf after normalization. Replacing...")
        feature_matrix_scaled = np.nan_to_num(feature_matrix_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    
    feature_tensor = torch.FloatTensor(feature_matrix_scaled)
    
    # Verify tensor is clean
    if torch.isnan(feature_tensor).any() or torch.isinf(feature_tensor).any():
        print("Warning: Tensor still has NaN/inf. Replacing with zeros...")
        feature_tensor = torch.nan_to_num(feature_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Create datasets
    train_dataset = TransitionDataset(train, user_to_idx, node_features_dict)
    val_dataset = TransitionDataset(val, user_to_idx, node_features_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = RoleTransitionPredictor(
        node_feat_dim=feature_dim,
        hidden_dim=128,
        num_roles=5,
        num_layers=2
    ).to(device)
    
    edge_index = edge_index.to(device)
    feature_tensor = feature_tensor.to(device)
    
    # Loss and optimizer (reduced learning rate to prevent NaN)
    class_weights = torch.tensor([0.1, 1.0, 1.0, 2.0, 3.0]).to(device)
    # Normalize weights to prevent extreme values
    class_weights = class_weights / class_weights.sum() * 5.0
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Reduced LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    num_epochs = 1
    best_val_f1 = 0.0
    
    print("\nTraining model...")
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0.0
        
        train_batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                                leave=False, unit="batch")
        for batch in train_batch_pbar:
            # Convert to tensors properly
            if isinstance(batch['user_idx'], torch.Tensor):
                user_indices = batch['user_idx'].long().to(device)
            else:
                user_indices = torch.tensor(batch['user_idx'], dtype=torch.long).to(device)
            
            if isinstance(batch['current_role'], torch.Tensor):
                current_roles = batch['current_role'].long().to(device)
            else:
                current_roles = torch.tensor(batch['current_role'], dtype=torch.long).to(device)
            
            if isinstance(batch['next_role'], torch.Tensor):
                next_roles = batch['next_role'].long().to(device)
            else:
                next_roles = torch.tensor(batch['next_role'], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            
            logits = model(feature_tensor, edge_index, user_indices, current_roles)
            
            # Check for NaN in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN/inf in logits. Skipping batch.")
                continue
            
            loss = criterion(logits, next_roles.long())
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/inf loss. Skipping batch.")
                continue
            
            loss.backward()
            
            # More aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print(f"Warning: NaN/inf gradients. Skipping update.")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        val_batch_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                              leave=False, unit="batch")
        with torch.no_grad():
            for batch in val_batch_pbar:
                # Convert to tensors properly
                if isinstance(batch['user_idx'], torch.Tensor):
                    user_indices = batch['user_idx'].long().to(device)
                else:
                    user_indices = torch.tensor(batch['user_idx'], dtype=torch.long).to(device)
                
                if isinstance(batch['current_role'], torch.Tensor):
                    current_roles = batch['current_role'].long().to(device)
                else:
                    current_roles = torch.tensor(batch['current_role'], dtype=torch.long).to(device)
                
                if isinstance(batch['next_role'], torch.Tensor):
                    next_roles = batch['next_role'].long().to(device)
                else:
                    next_roles = torch.tensor(batch['next_role'], dtype=torch.long).to(device)
                
                # Handle users not in graph
                valid_mask = user_indices < num_nodes
                if valid_mask.sum() > 0:
                    valid_indices = user_indices[valid_mask]
                    valid_roles = current_roles[valid_mask]
                    logits = model(feature_tensor, edge_index, valid_indices, valid_roles)
                    preds_valid = torch.argmax(logits, dim=1)
                    
                    # Create predictions for all users
                    preds = torch.full((len(user_indices),), 2, dtype=torch.long).to(device)  # Default: Contributor
                    preds[valid_mask] = preds_valid
                else:
                    # All users are new, use default prediction
                    preds = torch.full((len(user_indices),), 2, dtype=torch.long).to(device)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(next_roles.cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        train_loss /= len(train_loader)
        
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pt')
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_f1': f'{val_f1:.4f}',
                'best_f1': f'{best_val_f1:.4f}',
                'status': '✓ Saved'
            })
        else:
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_f1': f'{val_f1:.4f}',
                'best_f1': f'{best_val_f1:.4f}'
            })
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    
    val_preds = []
    val_labels = []
    val_current_roles = []
    
    print("\nEvaluating best model on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", unit="batch"):
            # Convert to tensors properly
            if isinstance(batch['user_idx'], torch.Tensor):
                user_indices = batch['user_idx'].long().to(device)
            else:
                user_indices = torch.tensor(batch['user_idx'], dtype=torch.long).to(device)
            
            if isinstance(batch['current_role'], torch.Tensor):
                current_roles = batch['current_role'].long().to(device)
            else:
                current_roles = torch.tensor(batch['current_role'], dtype=torch.long).to(device)
            
            if isinstance(batch['next_role'], torch.Tensor):
                next_roles = batch['next_role'].long().to(device)
            else:
                next_roles = torch.tensor(batch['next_role'], dtype=torch.long).to(device)
            
            # Handle users not in graph
            valid_mask = user_indices < num_nodes
            if valid_mask.sum() > 0:
                valid_indices = user_indices[valid_mask]
                valid_roles = current_roles[valid_mask]
                logits = model(feature_tensor, edge_index, valid_indices, valid_roles)
                preds_valid = torch.argmax(logits, dim=1)
                
                # Create predictions for all users
                preds = torch.full((len(user_indices),), 2, dtype=torch.long).to(device)  # Default: Contributor
                preds[valid_mask] = preds_valid
            else:
                # All users are new, use default prediction
                preds = torch.full((len(user_indices),), 2, dtype=torch.long).to(device)
            
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(next_roles.cpu().numpy())
            val_current_roles.extend(current_roles.cpu().numpy())
    
    # Compute metrics
    macro_f1 = f1_score(val_labels, val_preds, average='macro')
    print(f"\n{'='*60}")
    print(f"Final Validation Macro-F1 Score: {macro_f1:.4f}")
    print(f"{'='*60}")
    
    # Classification report
    unique_classes = sorted(set(val_labels) | set(val_preds))
    role_names = ['Inactive', 'Novice', 'Contributor', 'Expert', 'Moderator']
    print("\nClassification Report:")
    print(classification_report(
        val_labels, val_preds,
        labels=unique_classes,
        target_names=[role_names[i] for i in unique_classes],
        zero_division=0
    ))
    
    # Transition-level analysis
    print("\nTransition-level analysis:")
    val_df = val.copy()
    val_df['predicted_role'] = val_preds
    val_df['true_transition'] = val_df['current_role'].astype(str) + '->' + val_df['next_role'].astype(str)
    val_df['pred_transition'] = val_df['current_role'].astype(str) + '->' + val_df['predicted_role'].astype(str)
    
    transition_counts = val_df['true_transition'].value_counts()
    rare_threshold = len(val_df) * 0.05
    rare_transitions = transition_counts[transition_counts < rare_threshold].index.tolist()
    
    print(f"\nRare transitions (< 5%): {len(rare_transitions)}")
    if rare_transitions:
        rare_mask = val_df['true_transition'].isin(rare_transitions)
        rare_f1 = f1_score(
            val_df.loc[rare_mask, 'next_role'],
            val_df.loc[rare_mask, 'predicted_role'],
            average='macro'
        )
        print(f"Macro-F1 on rare transitions: {rare_f1:.4f}")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    test_features = pd.read_parquet('./data/processed/test_features.parquet')
    
    # Map test user IDs to indices (use train mapping, add new users if needed)
    test_user_to_idx = user_to_idx.copy()
    for uid in test_features['user_id'].unique():
        if uid not in test_user_to_idx:
            test_user_to_idx[uid] = len(test_user_to_idx)
    
    # Prepare test features (use mean features from train if user not seen)
    test_node_features_dict = node_features_dict.copy()
    for uid in test_features['user_id'].unique():
        if uid not in test_node_features_dict:
            # Use default features (mean of all users)
            test_node_features_dict[uid] = np.mean(feature_matrix, axis=0)
    
    test_dataset = TransitionDataset(test_features, test_user_to_idx, test_node_features_dict)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    test_preds = []
    print("\nMaking predictions on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", unit="batch"):
            # Convert to tensors properly
            if isinstance(batch['user_idx'], torch.Tensor):
                user_indices = batch['user_idx'].long().to(device)
            else:
                user_indices = torch.tensor(batch['user_idx'], dtype=torch.long).to(device)
            
            if isinstance(batch['current_role'], torch.Tensor):
                current_roles = batch['current_role'].long().to(device)
            else:
                current_roles = torch.tensor(batch['current_role'], dtype=torch.long).to(device)
            
            # Handle users not in training graph (those with index >= num_nodes)
            valid_mask = user_indices < num_nodes
            if valid_mask.sum() == 0:
                # All users are new, use default prediction (Contributor = 2)
                preds = torch.full((len(user_indices),), 2, dtype=torch.long)
            else:
                # Use model for known users
                valid_indices = user_indices[valid_mask]
                valid_roles = current_roles[valid_mask]
                logits = model(feature_tensor, edge_index, valid_indices, valid_roles)
                preds_valid = torch.argmax(logits, dim=1)
                # For new users, predict most common class (Contributor = 2)
                preds = torch.full((len(user_indices),), 2, dtype=torch.long).to(device)
                preds[valid_mask] = preds_valid
            
            test_preds.extend(preds.cpu().numpy())
    
    # Create submission
    submission = pd.DataFrame({
        'user_id': test_features['user_id'],
        'snapshot_id': test_features['snapshot_id'],
        'predicted_role': test_preds
    })
    
    os.makedirs('./submissions', exist_ok=True)
    submission.to_csv('./submissions/team_sam_GraphSAGE+LSTM_1epoch_training.csv', index=False)
    print("Saved predictions to ./team_sam_GraphSAGE+LSTM_1epoch_training.csv")
    
    # Clean up model file
    if os.path.exists('best_model.pt'):
        os.remove('best_model.pt')
    
    return model

if __name__ == '__main__':
    train_model()

