"""
Advanced baseline model for Role Transition Prediction challenge.
Uses GAT (Graph Attention Networks) + Jump Knowledge pooling for better performance than GraphSAGE+LSTM.

Improvements over baseline_GraphSAGE+LSTM.py:
- GAT: Attention over neighbors (learns different importance for each neighbor)
- Jump Knowledge: Aggregates multi-scale representations from all GNN layers
- Temporal encoding: snapshot_id embedded to model temporal context
- Residual connections in GNN for better gradient flow

This model uses:
- adjacency_all.parquet, node_features_all.parquet, train.parquet, test_features.parquet
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.manual_seed(42)
np.random.seed(42)


class GATWithJK(nn.Module):
    """GAT with Jump Knowledge pooling and residual connections."""
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=3, heads=4, jk_mode='cat'):
        super().__init__()
        self.num_layers = num_layers
        self.jk_mode = jk_mode  # 'cat', 'max', or 'lstm'
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        # First layer
        self.layers.append(GATConv(in_feats, hidden_feats // heads, heads=heads))
        self.bns.append(nn.LayerNorm(hidden_feats))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_feats, hidden_feats // heads, heads=heads))
            self.bns.append(nn.LayerNorm(hidden_feats))
        self.layers.append(GATConv(hidden_feats, out_feats, heads=1))
        self.dropout = nn.Dropout(0.2)
        if jk_mode == 'lstm':
            self.jk_lstm = nn.LSTM(out_feats, out_feats, batch_first=True)

    def forward(self, x, edge_index):
        xs = []
        h = x
        for i, layer in enumerate(self.layers):
            h_res = h
            h = layer(h, edge_index)
            if i < len(self.bns):
                h = self.bns[i](h)
            if i < len(self.layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)
                if h.shape == h_res.shape:
                    h = h + h_res  # Residual
            xs.append(h)
        if self.jk_mode == 'cat':
            return torch.cat(xs, dim=-1)
        elif self.jk_mode == 'max':
            return torch.stack(xs, dim=0).max(dim=0)[0]
        else:
            stacked = torch.stack(xs, dim=1)
            _, (h_jk, _) = self.jk_lstm(stacked)
            return h_jk.squeeze(0)


class RoleTransitionPredictorGAT(nn.Module):
    """GAT + Jump Knowledge + temporal encoding for role transition prediction."""
    def __init__(self, node_feat_dim, hidden_dim=128, num_roles=5, num_layers=3, jk_out_dim=None):
        super().__init__()
        jk_out = jk_out_dim or hidden_dim * num_layers  # cat mode
        self.gat = GATWithJK(node_feat_dim, hidden_dim, hidden_dim, num_layers=num_layers, jk_mode='cat')
        self.snapshot_embed = nn.Embedding(64, 16)  # snapshot_id -> temporal embedding
        combined_dim = jk_out + node_feat_dim + 5 + 16  # gat_jk + raw feats + role onehot + snapshot
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_roles)
        )

    def forward(self, x, edge_index, user_ids, current_roles, snapshot_ids=None):
        graph_emb = self.gat(x, edge_index)
        user_emb = graph_emb[user_ids]
        user_feats = x[user_ids]
        role_onehot = F.one_hot(current_roles.long(), num_classes=5).float()
        if snapshot_ids is not None:
            snap_emb = self.snapshot_embed(snapshot_ids.clamp(0, 63))
        else:
            snap_emb = torch.zeros(user_ids.size(0), 16, device=x.device)
        combined = torch.cat([user_emb, user_feats, role_onehot, snap_emb], dim=-1)
        return self.classifier(combined)


class TransitionDataset(Dataset):
    def __init__(self, transitions_df, user_to_idx, node_features_dict):
        self.transitions = transitions_df.reset_index(drop=True)
        self.user_to_idx = user_to_idx
        self.node_features_dict = node_features_dict

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        row = self.transitions.iloc[idx]
        user_id = row['user_id']
        user_idx = self.user_to_idx.get(user_id, 0)
        features = self.node_features_dict.get(user_id, np.zeros(7, dtype=np.float32))
        current_role = row['current_role']
        next_role = row.get('next_role', -1)
        snapshot_id = row.get('snapshot_id', 0)
        return {
            'user_idx': user_idx,
            'current_role': current_role,
            'next_role': next_role,
            'features': features,
            'snapshot_id': snapshot_id
        }


def build_graph_from_adjacency(user_to_idx, snapshot_id=None, max_edges=None):
    adjacency_all = pd.read_parquet('./data/processed/adjacency_all.parquet')
    if snapshot_id is not None:
        adjacency = adjacency_all[adjacency_all['snapshot_id'] == snapshot_id].copy()
    else:
        adjacency = adjacency_all.copy()
    edges = set()
    for _, row in adjacency.iterrows():
        src_uid, dst_uid = row['src'], row['dst']
        if src_uid in user_to_idx and dst_uid in user_to_idx:
            src_idx = user_to_idx[src_uid]
            dst_idx = user_to_idx[dst_uid]
            edges.add((src_idx, dst_idx))
            edges.add((dst_idx, src_idx))
    num_nodes = len(user_to_idx)
    for i in range(num_nodes):
        edges.add((i, i))
    if max_edges and len(edges) > max_edges:
        edges = set(list(edges)[:max_edges])
    if not edges:
        edges = [(i, i) for i in range(num_nodes)]
    src, dst = zip(*edges)
    edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    return edge_index, user_to_idx, {idx: uid for uid, idx in user_to_idx.items()}


def prepare_node_features(user_to_idx, snapshot_id=None):
    node_features_all = pd.read_parquet('./data/processed/node_features_all.parquet')
    if snapshot_id is not None:
        node_features = node_features_all[node_features_all['snapshot_id'] == snapshot_id].copy()
    else:
        node_features = node_features_all.copy()
    feature_cols = ['out_degree', 'in_degree', 'num_unique_recipients', 'num_unique_sources',
                    'total_interactions', 'activity_span_days', 'avg_interactions_per_day']
    user_features_agg = node_features.groupby('user_id')[feature_cols].mean().reset_index()
    node_features_dict = {}
    feature_matrix = []
    for uid in sorted(user_to_idx.keys()):
        user_feat = user_features_agg[user_features_agg['user_id'] == uid]
        features = user_feat[feature_cols].values[0].astype(np.float32) if len(user_feat) > 0 else np.zeros(len(feature_cols), dtype=np.float32)
        node_features_dict[uid] = features
        feature_matrix.append(features)
    return node_features_dict, np.array(feature_matrix, dtype=np.float32)


def load_data():
    train_full = pd.read_parquet('./data/processed/train.parquet')
    train_full['snapshot_start'] = pd.to_datetime(train_full['snapshot_start'])
    train_full = train_full.sort_values('snapshot_start')
    cutoff_idx = int(len(train_full) * 0.8)
    cutoff_date = train_full.iloc[cutoff_idx]['snapshot_start']
    train = train_full[train_full['snapshot_start'] < cutoff_date].copy()
    val = train_full[train_full['snapshot_start'] >= cutoff_date].copy()
    return train, val


def collate_fn(batch):
    return {
        'user_idx': torch.tensor([b['user_idx'] for b in batch], dtype=torch.long),
        'current_role': torch.tensor([b['current_role'] for b in batch], dtype=torch.long),
        'next_role': torch.tensor([b['next_role'] for b in batch], dtype=torch.long),
        'features': torch.tensor(np.stack([b['features'] for b in batch]), dtype=torch.float32),
        'snapshot_id': torch.tensor([b['snapshot_id'] for b in batch], dtype=torch.long)
    }


def train_model():
    print("Loading data...")
    train, val = load_data()
    print(f"Train: {len(train)}, Val: {len(val)}")

    all_users = set(train['user_id'].unique()) | set(val['user_id'].unique())
    user_to_idx = {uid: idx for idx, uid in enumerate(sorted(all_users))}
    num_nodes = len(user_to_idx)
    new_users = set(val['user_id'].unique()) - set(train['user_id'].unique())

    print("Building graph...")
    edge_index, _, _ = build_graph_from_adjacency(user_to_idx)
    print(f"Graph: {num_nodes} nodes, {edge_index.shape[1]} edges")

    print("Loading node features...")
    node_features_dict, feature_matrix = prepare_node_features(user_to_idx)
    default_feat = np.mean(feature_matrix, axis=0) if len(feature_matrix) > 0 else np.zeros(7, dtype=np.float32)
    for uid in new_users:
        if uid not in node_features_dict:
            node_features_dict[uid] = default_feat

    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    feature_matrix_scaled = np.nan_to_num(feature_matrix_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    feature_tensor = torch.FloatTensor(feature_matrix_scaled)

    train_dataset = TransitionDataset(train, user_to_idx, node_features_dict)
    val_dataset = TransitionDataset(val, user_to_idx, node_features_dict)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dim = feature_matrix.shape[1]
    jk_out = 128 * 3  # hidden * num_layers (cat mode)
    model = RoleTransitionPredictorGAT(
        node_feat_dim=feature_dim,
        hidden_dim=128,
        num_roles=5,
        num_layers=3,
        jk_out_dim=jk_out
    ).to(device)
    edge_index = edge_index.to(device)
    feature_tensor = feature_tensor.to(device)

    class_weights = torch.tensor([0.1, 1.0, 1.0, 2.0, 3.0]).to(device)
    class_weights = class_weights / class_weights.sum() * 5.0
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    num_epochs = 10
    best_val_f1 = 0.0

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            user_indices = batch['user_idx'].to(device)
            current_roles = batch['current_role'].to(device)
            next_roles = batch['next_role'].to(device)
            snapshot_ids = batch['snapshot_id'].to(device)
            optimizer.zero_grad()
            logits = model(feature_tensor, edge_index, user_indices, current_roles, snapshot_ids)
            loss = criterion(logits, next_roles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                user_indices = batch['user_idx'].to(device)
                current_roles = batch['current_role'].to(device)
                next_roles = batch['next_role'].to(device)
                snapshot_ids = batch['snapshot_id'].to(device)
                valid_mask = user_indices < num_nodes
                if valid_mask.sum() > 0:
                    logits = model(feature_tensor, edge_index, user_indices[valid_mask],
                                  current_roles[valid_mask], snapshot_ids[valid_mask])
                    preds_valid = torch.argmax(logits, dim=1)
                    preds = torch.full((len(user_indices),), 2, dtype=torch.long, device=device)
                    preds[valid_mask] = preds_valid
                else:
                    preds = torch.full((len(user_indices),), 2, dtype=torch.long, device=device)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(next_roles.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        scheduler.step(val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model_gat.pt')

    model.load_state_dict(torch.load('best_model_gat.pt'))
    model.eval()
    print(f"\nBest Validation Macro-F1: {best_val_f1:.4f}")

    # Test predictions
    print("\nMaking test predictions...")
    test_features = pd.read_parquet('./data/processed/test_features.parquet')
    test_user_to_idx = user_to_idx.copy()
    for uid in test_features['user_id'].unique():
        if uid not in test_user_to_idx:
            test_user_to_idx[uid] = len(test_user_to_idx)
    test_node_features_dict, _ = prepare_node_features(test_user_to_idx)
    for uid in test_features['user_id'].unique():
        if uid not in test_node_features_dict:
            test_node_features_dict[uid] = default_feat

    test_dataset = TransitionDataset(test_features, test_user_to_idx, test_node_features_dict)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)

    test_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            user_indices = batch['user_idx'].to(device)
            current_roles = batch['current_role'].to(device)
            snapshot_ids = batch['snapshot_id'].to(device)
            valid_mask = user_indices < num_nodes
            if valid_mask.sum() > 0:
                logits = model(feature_tensor, edge_index, user_indices[valid_mask],
                              current_roles[valid_mask], snapshot_ids[valid_mask])
                preds_valid = torch.argmax(logits, dim=1)
                preds = torch.full((len(user_indices),), 2, dtype=torch.long, device=device)
                preds[valid_mask] = preds_valid
            else:
                preds = torch.full((len(user_indices),), 2, dtype=torch.long, device=device)
            test_preds.extend(preds.cpu().numpy())

    submission = pd.DataFrame({
        'user_id': test_features['user_id'],
        'snapshot_id': test_features['snapshot_id'],
        'predicted_role': test_preds
    })
    os.makedirs('./submissions', exist_ok=True)
    submission.to_csv('./submissions/challenge_submission.csv', index=False)
    print("Saved to ./submissions/challenge_submission.csv")
    if os.path.exists('best_model_gat.pt'):
        os.remove('best_model_gat.pt')
    return model


if __name__ == '__main__':
    train_model()
