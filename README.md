# GNN Challenge: Role Transition Prediction in Temporal Networks

## 🎯 Challenge Overview

Welcome to the **Role Transition Prediction Challenge**! This competition focuses on predicting how user roles evolve over time in the Super User Stack Exchange temporal network.

### Problem Description

The task is to predict **role transitions** - how a user's role evolves from one temporal snapshot to the next. Given a user's current role and their interaction history up to time `t`, predict their role at time `t+k`.

**Roles:**
- **0 - Inactive**: User with minimal or no activity
- **1 - Novice**: User who primarily asks questions (high in-degree, low out-degree)
- **2 - Contributor**: User with balanced question-answer activity
- **3 - Expert**: Highly active user who provides many answers (high out-degree)
- **4 - Moderator**: Very active user who helps many different users (high activity, high unique recipients)

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

### Downloading the Dataset

**Important**: You must download the preprocessed dataset files and place them in the `data/processed/` directory before running any models.

**Required files to download:**
- `train.parquet` - Training set with labels (~1.5M rows)
- `test_features.parquet` - Test set features without labels (~700K rows)

**Directory structure after download:**
```
data/
└── processed/
    ├── train.parquet          # Download and place here
    └── test_features.parquet  # Download and place here
```

Make sure both files are in `data/processed/` before proceeding with the baselines or your own models.

### Data Format

All data files are stored in Parquet format in the `data/processed/` directory.

#### File Overview

| File | Rows | Description | Availability | Location |
|------|------|-------------|--------------|----------|
| `train.parquet` | ~1,476,626 | Training transitions (2009-2014) | Public | `data/processed/` (download required) |
| `test_features.parquet` | ~698,639 | Test features without labels (2014-2016) | Public | `data/processed/` (download required) |
| `test.parquet` | ~698,639 | Test ground truth with labels (for scoring) | Private (organizers only) | `data/private/` |

#### Column Descriptions

**Role Labels:**
- `0` = Inactive
- `1` = Novice  
- `2` = Contributor
- `3` = Expert
- `4` = Moderator

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
- ❌ `next_role` (to be predicted)
- ❌ `transition_label` (to be predicted)
- ❌ `next_snapshot_start` (future information)
- ❌ `next_snapshot_end` (future information)

**Available columns:**
- ✅ `user_id`, `snapshot_id`, `current_role`
- ✅ `snapshot_start`, `snapshot_end`
- ✅ All 8 graph features (out_degree, in_degree, etc.)

#### test.parquet (16 columns) - PRIVATE

Same structure as `train.parquet` with all columns including labels. This file is **only available to organizers** for scoring submissions.

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

## 🚀 Getting Started

### 1. Setup Environment

```bash
pip install -r starter_code/requirements.txt
```

### 2. Download Dataset

Download the preprocessed dataset files and place them in the `data/processed/` directory:

```
data/processed/
├── train.parquet          # Training set (required)
└── test_features.parquet  # Test features (required)
```

**Note**: The dataset files must be downloaded separately and placed in `data/processed/` before proceeding.

### 3. Run Baseline

**Simple Baseline (Random Forest):**
```bash
cd starter_code
python baseline.py
```

The simple baseline uses a Random Forest classifier on graph-based features. Expected performance:
- Overall Macro-F1: ~0.35-0.45
- Rare Transitions F1: ~0.15-0.25

**Advanced Baseline (GNN + LSTM):**
```bash
cd starter_code
python baseline2.py
```

The advanced baseline uses GraphSAGE (DGL) + LSTM to capture both graph structure and temporal dynamics. Expected performance:
- Overall Macro-F1: ~0.45-0.55
- Rare Transitions F1: ~0.25-0.35

This demonstrates how GNN methods can improve performance over simple feature-based models.

### 4. Make Predictions

Your submission should be a CSV file with columns:
- `user_id`: User identifier
- `snapshot_id`: Snapshot identifier
- `predicted_role`: Predicted next role (0-4)

Example:
```csv
user_id,snapshot_id,predicted_role
123,5,2
456,5,3
789,6,1
```

Save your submission to `submissions/your_submission.csv`

### 5. Score Your Submission

```bash
python scoring_script.py submissions/your_submission.csv
```

## 📋 Constraints

To ensure fair competition and focus on GNN methods:

1. **No External Data**: You cannot use external datasets, pre-trained embeddings, or any data not derived from the provided temporal graph.

2. **DGL Methods Only**: Use methods covered in DGL lectures 1.1-4.6:
   - Message passing (GCN, GraphSAGE, GAT, etc.)
   - Sampling methods (neighbor sampling, layer-wise sampling)
   - Graph construction and batching
   - Temporal graph methods

3. **Graph Features Only**: All features must be extracted from the temporal graph structure. No text features, user profiles, or external metadata.

4. **Temporal Split**: Respect the temporal split - do not use future information to predict past transitions.

5. **No Pre-trained Models**: Pre-trained models (e.g., BERT embeddings) are not allowed unless they are trained solely on the provided data.

## 🏆 Tips for Success

1. **Leverage Temporal Structure**: Use temporal GNN architectures (TGN, EvolveGCN) or sequence models (LSTM, Transformer) to capture temporal dynamics.

2. **Handle Class Imbalance**: Use techniques like class weighting, focal loss, or resampling to handle imbalanced role distributions.

3. **Focus on Rare Transitions**: Design your model to specifically improve performance on rare transitions, not just overall accuracy.

4. **Graph Sampling**: Use efficient sampling strategies (GraphSAGE, FastGCN) to handle the large graph size.

5. **Feature Engineering**: Extract rich temporal and structural features from the graph:
   - Temporal motifs
   - Neighborhood evolution patterns
   - Centrality measures over time
   - Interaction type distributions

6. **Ensemble Methods**: Combine multiple models or use ensemble techniques to improve robustness.

## 📁 Repository Structure

```
gnn-challenge/
├── .github/
│   └── workflows/
│       ├── evaluate_submission.yml    # Auto-evaluate PR submissions
│       └── update_leaderboard.yml     # Auto-update leaderboard
├── data/
│   ├── processed/                # Download dataset files here
│   │   ├── train.parquet         # Training set (download required)
│   │   └── test_features.parquet # Test features (download required)
│   └── private/                  # Private test labels (organizers only)
│       └── test.parquet          # Test ground truth (for scoring)
├── scripts/
│   ├── evaluate_all_submissions.py    # Evaluate all submissions
│   └── generate_leaderboard.py        # Generate leaderboard HTML/JSON
├── submissions/
│   └── sample_submission*.csv    # Example submission format
├── starter_code/
│   ├── baseline.py               # Simple baseline model (Random Forest)
│   ├── baseline2.py              # Advanced baseline model (GNN + LSTM)
│   └── requirements.txt          # Python dependencies
├── prepare_data.py               # Data preparation script
├── scoring_script.py             # Evaluation script
├── update_leaderboard.py         # Leaderboard update script
├── leaderboard.json              # Leaderboard data (auto-generated)
├── leaderboard.html              # Leaderboard page (auto-generated)
├── GITHUB_SETUP.md               # GitHub setup guide
└── README.md                     # This file
```

## 📝 Submission Guidelines

1. **Format**: CSV file with columns `user_id`, `snapshot_id`, `predicted_role`
2. **Naming**: Use descriptive names like `submissions/team_name_model_v1.csv`
3. **Validation**: Ensure all user_id and snapshot_id pairs from test set are included
4. **Predictions**: `predicted_role` must be integers 0-4

## 🔬 Baseline Models Details

**Simple Baseline** (`starter_code/baseline.py`):
- Uses Random Forest with 100 trees
- Features: current role + graph statistics (degrees, interactions, etc.)
- Handles class imbalance with `class_weight='balanced'`
- Expected performance: ~0.35-0.45 Macro-F1 overall, ~0.15-0.25 on rare transitions

**Advanced Baseline** (`starter_code/baseline2.py`):
- Uses GraphSAGE (DGL) for learning node embeddings from graph structure
- LSTM layer for capturing temporal dynamics
- Combines graph embeddings with node features and current role
- Handles class imbalance with weighted cross-entropy loss
- Expected performance: ~0.45-0.55 Macro-F1 overall, ~0.25-0.35 on rare transitions

**Your goal**: Significantly outperform both baselines using state-of-the-art GNN methods!

## 📚 References

- **Dataset**: [SNAP Super User Network](https://snap.stanford.edu/data/sx-superuser.html)
- **GNNs**: [Basira Lab youtube](https://www.youtube.com/playlist?list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T)
- **Tutorials GNNs**: [Basira Lab Github](https://github.com/basiralab/dgl)

## 🏆 Leaderboard

The leaderboard is automatically updated when you submit your solution via Pull Request.

👉 **[View Live Leaderboard](leaderboard.html)** (or check `leaderboard.html` in the repository)

The leaderboard shows:
- **Rank**: Your position based on Weighted Macro-F1 score
- **Team Name**: Your submission filename (without .csv)
- **Weighted Macro-F1**: Primary evaluation metric
- **Overall Macro-F1**: Overall performance across all transitions
- **Rare Transitions F1**: Performance on rare transitions (< 5% frequency)

## 🤝 How to Submit

### Submission Process

1. **Fork this repository** to your GitHub account

2. **Download the dataset** and place files in `data/processed/`:
   - `train.parquet`
   - `test_features.parquet`

3. **Create your model** using the starter code or your own implementation

4. **Generate predictions** for the test set and save as CSV:
   ```bash
   # Your submission file should be named: submissions/your_team_name.csv
   # Format: user_id, snapshot_id, predicted_role
   ```

5. **Create a Pull Request** with your submission file:
   - Add your CSV file to `submissions/your_team_name.csv`
   - The GitHub Action will automatically evaluate your submission
   - If valid, your score will appear on the leaderboard

### Submission Requirements

- **File naming**: `submissions/team_name.csv` (use your team name)
- **CSV format**: Must have columns `user_id`, `snapshot_id`, `predicted_role`
- **Predictions**: `predicted_role` must be integers 0-4
- **Completeness**: Must include predictions for all test samples

### Automatic Evaluation

When you submit a Pull Request:
1. GitHub Actions automatically runs the evaluation
2. Your submission is scored using the test set
3. Results are posted as a comment on your PR
4. If your score is valid, the leaderboard is updated automatically
5. The leaderboard HTML page is regenerated and available on GitHub Pages

### Local Testing

Before submitting, test your submission locally:
```bash
python scoring_script.py submissions/your_team_name.csv
```

This will show you the same metrics that will be used for the leaderboard.

## 📄 License

See LICENSE file for details.

---

**Good luck and happy modeling! 🚀**

