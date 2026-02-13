# SHIFT-GNN

**Structural & Historical Inference of Functional role Transitions with GNNs**

*Predicting role shifts in temporal networks*

## 🎯 Challenge Overview

Welcome to the **SHIFT-GNN Challenge**! This competition focuses on predicting how user roles evolve over time in the Super User Stack Exchange temporal network.

**[🏆 View SHIFT-GNN Leaderboard](https://samuelmatia.github.io/shift-gnn-challenge/leaderboard.html)**

### Problem Description

The task is to predict **role transitions** - how a user's role evolves from one temporal snapshot to the next. Given a user's current role and their interaction history up to time `t`, predict their role at time `t+k`.

**Roles:**
- **0 - Inactive**: User with minimal or no activity
- **1 - Novice**: User who primarily asks questions (high in-degree, low out-degree)
- **2 - Contributor**: User with balanced question-answer activity
- **3 - Expert**: Highly active user who provides many answers (high out-degree)
- **4 - Moderator**: Very active user who helps many different users (high activity, high unique recipients)

Roles are assigned using a fixed heuristic based on user activity and interaction patterns (degrees, activity spans), and remain consistent across all temporal snapshots. They are not human-annotated ground truth, but deterministic proxy labels.

### Formal Setup

Let $G_t = (V, E_t)$ be the temporal interaction graph at time $t$. Each node $v \in V$ has a role $y_v^t \in \{0, \ldots, 4\}$. Given the interaction history $H_v^t$ of user $v$ up to time $t$, the task is to predict $y_v^{t+k}$ (the role at the next snapshot).

### What's Challenging ? 

1. **Temporal Shift**: Training data comes from 2009-2013, validation from 2013-2014, and test from 2014-2016. The distribution of user behaviors and network patterns changes over time, making generalization challenging.

2. **Rare Transitions**: The evaluation metric focuses on rare transitions (e.g., "Novice → Expert", "Contributor → Inactive"), which are the most valuable to predict but occur infrequently.

3. **Class Imbalance**: Role distributions are highly imbalanced, with the vast majority of users being Inactive (~78% of current roles, ~88% of next roles), followed by Contributors (~9%) and Novices (~8.5%).

4. **Graph Structure**: Features must be extracted from the temporal graph structure alone - no external data or text features are allowed.

5. **Temporal Dynamics**: The challenge requires modeling both the temporal evolution of individual users and the network-level dynamics.


## 📊 Dataset

The dataset is based on the **Super User Stack Exchange temporal network** from SNAP:
- **Nodes**: Users (194,085 unique users)
- **Edges**: Temporal interactions (1,443,339 edges)
- **Time Span**: ~7 years (2009-2016)
- **Edge Types**: Answers to questions, comments to questions, comments to answers

**Files available in `data/processed/` :**
- `train.parquet` - Training set with labels (user_id, snapshot_id, current_role, next_role, timestamps)
- `test_features.parquet` - Test set without labels (user_id, snapshot_id, current_role, timestamps)
- `adjacency_all.parquet` - **Adjacency matrices A_t** (all edges with snapshot_id, COO sparse format) - **REQUIRED for GNN**
- `node_features_all.parquet` - **Node features X** (all node features with snapshot_id) - **REQUIRED for GNN**

**⚠️ IMPORTANT: GNN Requirement**

The challenge **requires graph neural networks (GNNs)**. The transition files (`train.parquet`, `test_features.parquet`) contain **only identifiers and roles** - no aggregated features. 

**To build your model, you MUST:**
1. Load `adjacency_all.parquet` to construct the graph structure A_t for each snapshot
2. Load `node_features_all.parquet` to get node features X_t (or compute your own from the graph)
3. Use GNN message passing to learn node representations from the graph structure
4. Predict role transitions using the learned representations



### Graph Specification

**Adjacency Matrix A_t**: For each snapshot t, the adjacency matrix A_t is provided in sparse COO (Coordinate) format:
- **File**: `data/processed/adjacency_all.parquet`
- **Columns**: `snapshot_id`, `src` (source node), `dst` (destination node)
- **Usage**: Filter rows where `snapshot_id == t` to get A_t for snapshot t
- **Format**: Sparse COO format - can be converted to dense matrix or CSR/CSR format for GNN libraries (PyG, DGL)

**Node Features X**: Each node v at snapshot t has a feature vector:
- **File**: `data/processed/node_features_all.parquet`
- **Columns**: `user_id`, `snapshot_id`, `out_degree`, `in_degree`, `num_unique_recipients`, `num_unique_sources`, `total_interactions`, `activity_span_days`, `avg_interactions_per_day`
- **Usage**: Filter rows where `snapshot_id == t` to get X_t (node features matrix) for snapshot t
- **Shape**: For snapshot t, X_t has shape [num_nodes, num_features]

**Example usage (PyTorch Geometric)**:
```python
import pandas as pd
import torch
from torch_geometric.data import Data

# Load adjacency matrices
adjacency = pd.read_parquet('data/processed/adjacency_all.parquet')
A_t = adjacency[adjacency['snapshot_id'] == 0]  # Get A_0

# Load node features
node_features = pd.read_parquet('data/processed/node_features_all.parquet')
X_t = node_features[node_features['snapshot_id'] == 0]  # Get X_0

# Create edge index for PyG (COO format: [2, num_edges])
edge_index = torch.tensor([
    A_t['src'].values,
    A_t['dst'].values
], dtype=torch.long)

# Create node features tensor
x = torch.tensor(X_t[['out_degree', 'in_degree', ...]].values, dtype=torch.float)

# Create PyG Data object
graph = Data(x=x, edge_index=edge_index)
# Now use this graph with your GNN model
```

**Example usage (DGL)**:
```python
import pandas as pd
import dgl
import torch

# Load and construct graph similarly
# See starter_code/baseline_GraphSAGE+LSTM.py for a complete example
```


### Graph Visualization

Below is a sample visualization of the temporal network structure:

![Graph Visualization](docs/graph_visualization.png)

*Sample visualization showing user interactions and role distributions in a temporal snapshot. Nodes represent users (colored by role), edges represent interactions, and node size indicates activity level.*



## 🎯 Evaluation Metric

**Primary Metric: Weighted Macro-F1 Score**

The primary evaluation metric uses **inverse frequency weighting**, giving more importance to rare transitions while still considering all transitions. This balances the need to predict rare but valuable transitions (e.g., "Novice → Expert", "Contributor → Inactive") with overall model performance.

The final score is computed as:
```
Score = Weighted Macro-F1 Score
       = Macro-F1 with sample weights = 1 / transition_frequency
```

This metric:
- Rewards accurate prediction of rare transitions (higher weight)
- Still considers common transitions (lower weight)
- Provides a balanced evaluation across all transition types

**Additional metrics reported:**
- Overall Macro-F1 (unweighted)
- Rare Transitions Macro-F1 (transitions occurring in < 5% of cases)
- Per-transition F1 scores


## 📋 Constraints

To ensure fair competition and focus on scalable GNN methods:

1. **GNN Required**  
   - The transition files (`train.parquet`, `test_features.parquet`) contain **only identifiers and roles** - no aggregated features.
   - You **MUST** use `adjacency_all.parquet` to construct graph structures and `node_features_all.parquet` for node features.

2. **No External Data**  
   Only the provided graph and features may be used.

3. **Graph Features Only**  
   No handcrafted features or external embeddings are allowed beyond what's provided in `node_features_all.parquet`.

4. **Train on CPU Only**  
   - Models must be trainable on a standard CPU environment.
   - Participants are encouraged to use **efficient training strategies** such as:
     - neighbor sampling 
     - subgraph or mini-batch training
     - memory-efficient message passing
   - Full-batch training on the entire graph is discouraged if it leads to excessive computation time or memory usage.

5. **One Submission Per Participant**  
   Only one submission attempt per participant is allowed and is enforced. Your first valid submission is recorded on the leaderboard; any later PR is evaluated for your information but does not update your score.


## 🤝 How to Submit

### Submission Process

1. **Fork this repository** to your GitHub account.

2. **Use the provided data**, located in `data/processed/`

3. **Build your model** using the starter code or your own implementation.

4. **Generate predictions** for the test set and save them as a CSV file with the required format:

   **Required columns:**
   - `user_id`: User identifier  
   - `snapshot_id`: Snapshot identifier  
   - `predicted_role`: Predicted next role (integer values from 0 to 4)

   **Example submission:**
```csv
   user_id,snapshot_id,predicted_role
   123,5,2
   456,5,3
   789,6,1 
```

**Put your submission in `submissions/`** with:
- `challenge_submission.csv` — your predictions
- `metadata.json` — records whether produced by a **human**, **LLM**, or **both** (`model_type`: `human` | `llm` | `human+llm`, optional `notes`)

5. **Submit via Google Form**

   **🔒 Private Submissions**: To keep submissions private, submit your file via the Google Form:
   
   **[👉 Submit Your Solution](LINK_TO_YOUR_GOOGLE_FORM)** *(Replace with your actual Google Form link)*
   
   **Required information:**
   - **Team Name**: Your team/participant name (will appear on leaderboard)
   - **Model Type**: `human`, `llm`, or `human+llm`
   - **Submission File**: Upload your `challenge_submission.csv` file
   
   **Important:**
   - ✅ Only **one submission per participant** is allowed (enforced by Google Form)
   - ✅ Your CSV file must have columns: `user_id`, `snapshot_id`, `predicted_role`
   - ✅ Submissions are processed periodically and scores appear on the public leaderboard
   - ✅ **Your CSV file remains private** - only scores and ranks are displayed publicly

6. **Check Your Score**

   After submission, your score will appear on the [leaderboard](leaderboard.html) within a few hours (or immediately if processed manually). Only your **team name**, **scores**, and **rank** are displayed publicly - your submission file is never visible to other participants.



## 🏆 SHIFT-GNN Leaderboard

👉 **[View SHIFT-GNN Leaderboard](https://samuelmatia.github.io/shift-gnn-challenge/leaderboard.html)**

The interactive leaderboard shows:
- **Rank**, **Team** (GitHub username), **Weighted Macro-F1** (primary metric), **Overall Macro-F1**, **Rare Transitions F1**
- **Model Type**: human, llm, or human+llm (from `metadata.json`)
- **Notes**: optional notes from metadata
- **Submission Time**





## 📚 References

- **Dataset**: [SNAP Super User Network](https://snap.stanford.edu/data/sx-superuser.html)
- **GNNs**: [Basira Lab youtube](https://www.youtube.com/playlist?list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T)
- **Tutorials GNNs**: [Basira Lab Github](https://github.com/basiralab/dgl)


## 📄 License

See LICENSE file for details.

---

**Good luck and happy modeling! 🚀**

