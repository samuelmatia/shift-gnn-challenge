# SHIFT-GNN Challenge

**Structural & Historical Inference of Functional role Transitions with GNNs**

*Predicting role shifts in temporal networks*

## 🎯 Challenge Overview

Welcome to the **SHIFT-GNN Challenge**! This competition focuses on predicting how user roles evolve over time in the Super User Stack Exchange temporal network.

**[🏆 View SHIFT-GNN Leaderboard](https://samuelmatia.github.io/shift-gnn-challenge/leaderboard.html)**

**[👉 Submit Your Solution](https://docs.google.com/forms/d/e/1FAIpQLScIvsMOmkAv5KFxz57GBiGBqo37SwHSVIrxQ92drUxp35jLCw/viewform)**

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
  - **Size**: 490,957 samples
  - **Time period**: 2009-2013
- `test_features.parquet` - Test set without labels (user_id, snapshot_id, current_role, timestamps)
  - **Size**: 186,158 samples
  - **Time period**: 2014-2016
  - **⚠️ This is what you need to predict!**
- `adjacency_all.parquet` - **Adjacency matrices A_t** (all edges with snapshot_id, COO sparse format) 
- `node_features_all.parquet` - **Node features X** (all node features with snapshot_id) 

The challenge **requires graph neural networks (GNNs)**


**To build your model, you MUST:**
1. Load `adjacency_all.parquet` to construct the graph structure A_t for each snapshot
2. Load `node_features_all.parquet` to get node features X_t (or compute your own from the graph)
3. Use GNN message passing to learn node representations from the graph structure
4. Predict role transitions using the learned representations



### Graph Specification

**Adjacency Matrix A_t**: For each snapshot t, the adjacency matrix A_t is provided in sparse COO (Coordinate) format.
**Node Features X**: Each node v at snapshot t has a feature vector.

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


## 📋 Constraints

To ensure fair competition and focus on scalable GNN methods:

1. **GNN Required**  

2. **No External Data**  
   Only the provided graph and features may be used.

3. **Train on CPU Only**  
   - Models must be trainable on a standard CPU environment.
   - Participants are encouraged to use **efficient training strategies** such as:
     - neighbor sampling 
     - subgraph or mini-batch training
     - memory-efficient message passing
   - Full-batch training on the entire graph is discouraged if it leads to excessive computation time or memory usage.

5. **One Submission Per Participant**  


## 🤝 How to Submit

### Submission Process

1. **Clone or download this repository** to access the data and starter code.

2. **Use the provided data**, located in `data/processed/`

3. **Build your model** using the starter code (`starter_code/`) or your own implementation.

4. **Generate predictions** for the test set:

   **Test data file**: `data/processed/test_features.parquet`
   - Contains: `user_id`, `snapshot_id`, `current_role`, `timestamps`
   - **Size**: 186,158 samples (you must predict `next_role` for all of them)
   - **No labels** - this is what you need to predict!
   - Your model should predict `next_role` for each `(user_id, snapshot_id)` pair in this file
   
   **Prediction process:**
   - Load `test_features.parquet` to get the test instances
   - For each `(user_id, snapshot_id)` pair, predict the `next_role`
   - Save predictions as a CSV file with the required format

   **Required CSV format:**
   - **Columns**: `user_id`, `snapshot_id`, `predicted_role`
   - `user_id`: User identifier (must match `user_id` from `test_features.parquet`)
   - `snapshot_id`: Snapshot identifier (must match `snapshot_id` from `test_features.parquet`)
   - `predicted_role`: Predicted next role (integer values: 0, 1, 2, 3, or 4)

   **Example submission (`challenge_submission.csv`):**
```csv
user_id,snapshot_id,predicted_role
123,5,2
456,5,3
789,6,1
```

   **File requirements:**
   - Filename: `challenge_submission.csv` (or any name ending in `.csv`)
   - Format: CSV with UTF-8 encoding
   - Size: Maximum 100 MB
   - All three columns must be present: `user_id`, `snapshot_id`, `predicted_role`
   - Must include predictions for **all 186,158** `(user_id, snapshot_id)` pairs in `test_features.parquet`
   - Your CSV should have exactly 186,158 rows (plus header)



5. **Submit via Google Form**

   **🔒 Private Submissions**: To keep submissions private and ensure fair competition, submit your file via the Google Form:
   
   **[👉 Submit Your Solution](https://docs.google.com/forms/d/e/1FAIpQLScIvsMOmkAv5KFxz57GBiGBqo37SwHSVIrxQ92drUxp35jLCw/viewform)**
   
   **Required information in the form:**
   - **Team Name** ⭐: Your team/participant name (will appear on the public leaderboard)
   - **Model Type** ⭐: Select one:
     - `human` - Model developed entirely by human(s)
     - `llm` - Model developed using LLM assistance (e.g., ChatGPT, Claude, etc.)
     - `human+llm` - Collaborative development between human(s) and LLM(s)
   - **Submission File** ⭐: Upload your `challenge_submission.csv` file (max 100 MB)
   
   **Important notes:**
   - Only **one submission per participant** is allowed (enforced automatically by the system)
   - Your CSV file must have exactly these columns: `user_id`, `snapshot_id`, `predicted_role`
   - File format: CSV with UTF-8 encoding
   - Make sure your predictions are integers from 0 to 4 (valid role values)

7. **Check Your Score**

   After submission, your score will appear on the [leaderboard](https://samuelmatia.github.io/shift-gnn-challenge/leaderboard.html) automatically. Only your **team name**, **scores**, and **rank** are displayed publicly - your submission file is never visible to other participants.
   


## 🏆 SHIFT-GNN Leaderboard

👉 **[View SHIFT-GNN Leaderboard](https://samuelmatia.github.io/shift-gnn-challenge/leaderboard.html)**


## 📚 References

- **Dataset**: [SNAP Super User Network](https://snap.stanford.edu/data/sx-superuser.html)
- **GNNs**: [Basira Lab youtube](https://www.youtube.com/playlist?list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T)
- **Tutorials GNNs**: [Basira Lab Github](https://github.com/basiralab/dgl)


## 📄 License

See LICENSE file for details.

---

**Good luck and happy modeling! 🚀**

