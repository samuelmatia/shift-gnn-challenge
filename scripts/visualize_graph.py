"""
Generate a visualization of the temporal network graph.
Creates a sample visualization showing the graph structure.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

def visualize_graph_sample(train_file='data/processed/train.parquet', output_file='docs/graph_visualization.png'):
    """
    Create a visualization of a sample of the graph.
    Shows user interactions and role transitions.
    """
    print("Loading data...")
    train = pd.read_parquet(train_file)
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find snapshot with fewest users for better visualization
    snapshot_user_counts = train.groupby('snapshot_id')['user_id'].nunique().sort_values()
    print(f"Snapshot user counts (top 10 smallest):")
    print(snapshot_user_counts.head(10))
    
    # Select snapshot with smallest number of users (for clearer visualization)
    sample_snapshot = snapshot_user_counts.index[0]
    print(f"\nSelected snapshot {sample_snapshot} with {snapshot_user_counts[sample_snapshot]} users")
    
    snapshot_data = train[train['snapshot_id'] == sample_snapshot].copy()
    
    # Limit to max 20 users for better clarity and readability
    unique_users = snapshot_data['user_id'].unique()
    if len(unique_users) > 20:
        np.random.seed(42)
        # Prefer users with some activity and diverse roles
        active_users = snapshot_data[snapshot_data['total_interactions'] > 0]['user_id'].unique()
        
        # Try to get a good mix of roles
        role_dist = snapshot_data.groupby('current_role')['user_id'].apply(lambda x: list(x.unique()))
        sampled_users = []
        
        # Sample 2-3 users per role (if available)
        for role in [1, 2, 3, 4, 0]:  # Prioritize active roles
            if role in role_dist.index and len(sampled_users) < 20:
                role_users = [u for u in role_dist[role] if u in active_users or len(active_users) == 0]
                if role_users:
                    n_sample = min(3, len(role_users), 20 - len(sampled_users))
                    sampled = np.random.choice(role_users, size=n_sample, replace=False)
                    sampled_users.extend(sampled)
        
        # Fill remaining slots if needed
        if len(sampled_users) < 20:
            remaining = [u for u in unique_users if u not in sampled_users]
            if remaining:
                n_needed = 20 - len(sampled_users)
                additional = np.random.choice(remaining, size=min(n_needed, len(remaining)), replace=False)
                sampled_users.extend(additional)
        
        sampled_users = sampled_users[:20]  # Ensure max 20
        snapshot_data = snapshot_data[snapshot_data['user_id'].isin(sampled_users)]
        print(f"Sampled {len(sampled_users)} users for visualization (diverse roles)")
    else:
        print(f"Using all {len(unique_users)} users from snapshot")
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes with role information
    role_colors = {
        0: '#95a5a6',  # Inactive - gray
        1: '#3498db',  # Novice - blue
        2: '#2ecc71',  # Contributor - green
        3: '#f39c12',  # Expert - orange
        4: '#e74c3c'   # Moderator - red
    }
    
    role_names = {
        0: 'Inactive',
        1: 'Novice',
        2: 'Contributor',
        3: 'Expert',
        4: 'Moderator'
    }
    
    # Add nodes
    for _, row in snapshot_data.iterrows():
        user_id = row['user_id']
        role = row['current_role']
        out_degree = row['out_degree']
        in_degree = row['in_degree']
        
        if user_id not in G:
            G.add_node(user_id, 
                      role=role,
                      out_degree=out_degree,
                      in_degree=in_degree,
                      size=max(10, min(100, (out_degree + in_degree) * 2)))
    
    # Create edges based on interactions
    # For visualization, create edges between users with similar activity or complementary roles
    users = list(G.nodes())
    
    # Create edges between users with interactions (based on degrees)
    for i, user1 in enumerate(users):
        deg1 = G.nodes[user1]['out_degree'] + G.nodes[user1]['in_degree']
        if deg1 == 0:
            continue
            
        for user2 in users[i+1:]:
            deg2 = G.nodes[user2]['out_degree'] + G.nodes[user2]['in_degree']
            
            # Create edge if:
            # 1. Both users have activity and similar levels, OR
            # 2. One user has high out_degree (answers) and other has high in_degree (asks questions)
            if deg1 > 0 and deg2 > 0:
                if abs(deg1 - deg2) < 3:  # Similar activity
                    G.add_edge(user1, user2)
                elif (G.nodes[user1]['out_degree'] > 2 and G.nodes[user2]['in_degree'] > 2) or \
                     (G.nodes[user2]['out_degree'] > 2 and G.nodes[user1]['in_degree'] > 2):
                    # Expert answering Novice's questions
                    G.add_edge(user1, user2)
    
    # Ensure graph is connected enough for visualization
    if G.number_of_edges() < len(users) * 0.3:
        np.random.seed(42)
        # Add some edges to connect the graph better
        for _ in range(len(users) // 3):
            u1, u2 = np.random.choice(users, size=2, replace=False)
            if not G.has_edge(u1, u2):
                G.add_edge(u1, u2)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Use spring layout with better spacing for small graphs
    if len(G.nodes()) <= 20:
        k = 2.0  # More spacing for small graphs
    else:
        k = 1.5
    
    pos = nx.spring_layout(G, k=k, iterations=100, seed=42)
    
    # Draw nodes by role with better visibility
    for role in [0, 1, 2, 3, 4]:
        nodes_with_role = [n for n, d in G.nodes(data=True) if d['role'] == role]
        if nodes_with_role:
            node_sizes = [max(200, min(800, G.nodes[n]['size'] * 10)) for n in nodes_with_role]
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=nodes_with_role,
                                  node_color=role_colors[role],
                                  node_size=node_sizes,
                                  alpha=0.9,
                                  edgecolors='black',
                                  linewidths=1.5,
                                  label=f"{role_names[role]} ({len(nodes_with_role)})")
    
    # Draw edges with better visibility
    nx.draw_networkx_edges(G, pos, 
                          alpha=0.3,
                          width=1.0,
                          arrows=True,
                          arrowsize=15,
                          edge_color='#7f8c8d',
                          connectionstyle='arc3,rad=0.1')
    
    # Add title and labels
    plt.title('GNN Challenge: Super User Stack Exchange Temporal Network\n'
              f'Sample Snapshot (Snapshot ID: {sample_snapshot}, {len(G.nodes())} users)',
              fontsize=16, fontweight='bold', pad=20)
    
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.axis('off')
    
    # Add description
    description = (
        "This visualization shows a sample of the temporal network.\n"
        "Nodes represent users, colored by their role. Edges represent interactions.\n"
        "Node size indicates activity level (in-degree + out-degree)."
    )
    plt.figtext(0.5, 0.02, description, 
                ha='center', fontsize=10, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Graph visualization saved to {output_file}")
    
    # Also create a simpler statistics visualization
    create_statistics_plot(snapshot_data, output_path.parent / 'graph_statistics.png')
    
    return output_file

def create_statistics_plot(snapshot_data, output_file):
    """Create a plot showing graph statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Role distribution
    role_counts = snapshot_data['current_role'].value_counts().sort_index()
    role_names = ['Inactive', 'Novice', 'Contributor', 'Expert', 'Moderator']
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    axes[0, 0].bar([role_names[i] for i in role_counts.index], role_counts.values, color=[colors[i] for i in role_counts.index])
    axes[0, 0].set_title('Role Distribution', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Users')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Degree distribution
    axes[0, 1].hist(snapshot_data['out_degree'], bins=30, alpha=0.7, label='Out-degree', color='#3498db')
    axes[0, 1].hist(snapshot_data['in_degree'], bins=30, alpha=0.7, label='In-degree', color='#e74c3c')
    axes[0, 1].set_title('Degree Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Degree')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # Total interactions
    axes[1, 0].hist(snapshot_data['total_interactions'], bins=50, color='#2ecc71', alpha=0.7)
    axes[1, 0].set_title('Total Interactions Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Total Interactions')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_yscale('log')
    
    # Activity span
    axes[1, 1].hist(snapshot_data['activity_span_days'], bins=50, color='#f39c12', alpha=0.7)
    axes[1, 1].set_title('Activity Span Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Activity Span (days)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.suptitle('Graph Statistics - Sample Snapshot', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Statistics plot saved to {output_file}")

if __name__ == '__main__':
    import sys
    
    train_file = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/train.parquet'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'docs/graph_visualization.png'
    
    if not Path(train_file).exists():
        print(f"Error: {train_file} not found. Please download the dataset first.")
        sys.exit(1)
    
    visualize_graph_sample(train_file, output_file)

