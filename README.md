# SHIFT-GNN

**Structural & Historical Inference of Functional role Transitions with GNNs**

*Predicting role shifts in temporal networks*

## 🎯 Challenge Overview

Welcome to the **SHIFT-GNN Challenge**! This competition focuses on predicting how user roles evolve over time in the Super User Stack Exchange temporal network.

**[🏆 View SHIFT-GNN Leaderboard](https://samuelmatia.github.io/gnn-role-transition-challenge/leaderboard.html)**

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
- `train.parquet` - Training set with labels 
- `test_features.parquet` - Test set features without labels

The provided features (`out_degree`, `in_degree`, etc.) are derived from the temporal graph structure. The challenge **requires graph neural networks (GNNs)** or graph-based models that exploit the full adjacency $E_t$; feature-only tabular models are not allowed.

#### Column Descriptions

#### train.parquet (16 columns)

| Column Name | Type | Description |
|-------------|------|-------------|
| `user_id` | int64 | Unique user identifier |
| `snapshot_id` | int64 | Temporal snapshot identifier (3-month windows) |
| `current_role` | int64 | Current role (0-4) |
| `next_role` | int64 | Next role to predict (target variable) |
| `transition_label` | string | Transition label (e.g., "2->0") |
| `snapshot_start` | datetime64[ns] | Start timestamp of current snapshot |
| `snapshot_end` | datetime64[ns] | End timestamp of current snapshot |
| `next_snapshot_start` | datetime64[ns] | Start timestamp of next snapshot |
| `next_snapshot_end` | datetime64[ns] | End timestamp of next snapshot |
| `out_degree` | int64 | Number of outgoing edges (answers/comments given) |
| `in_degree` | int64 | Number of incoming edges (questions/comments received) |
| `num_unique_recipients` | int64 | Number of unique users this user interacted with (outgoing) |
| `num_unique_sources` | int64 | Number of unique users who interacted with this user (incoming) |
| `total_interactions` | int64 | Total number of interactions in the snapshot |
| `activity_span_days` | int64 | Number of days between first and last interaction |
| `avg_interactions_per_day` | float64 | Average interactions per day (total_interactions / activity_span_days) |

#### test_features.parquet (12 columns)

Same structure as `train.parquet`, but **without** the following columns (labels are hidden):
-  `next_role` (to be predicted)
-  `transition_label` (to be predicted)
-  `next_snapshot_start` (future information)
-  `next_snapshot_end` (future information)


#### Example Data Row

```python
{
    'user_id': 34217,
    'snapshot_id': 7,
    'current_role': 2,           # Contributor
    'next_role': 0,              # Inactive (to predict)
    'transition_label': '2->0',  # Contributor -> Inactive
    'snapshot_start': Timestamp('2008-08-02 04:32:45'),
    'snapshot_end': Timestamp('2008-11-02 04:32:45'),
    'out_degree': 2,
    'in_degree': 2,
    'num_unique_recipients': 2,
    'num_unique_sources': 2,
    'total_interactions': 4,
    'activity_span_days': 51,
    'avg_interactions_per_day': 0.078
}
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

1. **No External Data**  
   Only the provided graph and features may be used.

2. **Graph Features Only**  
   No handcrafted features or external embeddings are allowed.

3. **Train on CPU Only**  
   - Models must be trainable on a standard CPU environment.
   - Participants are encouraged to use **efficient training strategies** such as:
     - neighbor sampling 
     - subgraph or mini-batch training
     - memory-efficient message passing
   - Full-batch training on the entire graph is discouraged if it leads to excessive computation time or memory usage.


## 🤝 How to Submit

### Submission Process

1. **Fork this repository** to your GitHub account.

2. **Use the provided data**, located in `data/processed/`:
   - `train.parquet`
   - `test_features.parquet`

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

5. **Score Your Submission**

```bash
python scoring_script.py submissions/challenge_submission.csv
```

6. **Create a Pull Request** with your submission:
   - Add `challenge_submission.csv` and `metadata.json` in **`submissions/`**.
   - Your **leaderboard name** is your **GitHub username** (one entry per participant; you can update by pushing new commits).
   - When a PR is opened or updated: it is validated, scored, and the leaderboard is updated automatically. The score is posted as a comment on the PR.



## 🏆 SHIFT-GNN Leaderboard

👉 **[View SHIFT-GNN Leaderboard](https://samuelmatia.github.io/gnn-role-transition-challenge/leaderboard.html)**

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

