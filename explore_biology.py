from openbiolink.evaluation.dataLoader import DataLoader
import torch
import numpy as np
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from itertools import islice

# Define relation mappings based on OpenBioLink documentation
RELATION_NAMES = {
    0: "ACTIVATED_BY",
    1: "ACTIVATES",
    2: "BINDS_TO",
    3: "DEACTIVATED_BY",
    4: "DEACTIVATES",
    5: "DECREASES_EXPRESSION",
    6: "DECREASES_MOLECULAR_MODIFICATION",
    7: "DECREASES_MUTATION_RATE",
    8: "DECREASES_REACTION",
    9: "DECREASES_SYNTHESIS",
    10: "EXPRESSED_IN",
    11: "HAS_ACTIVE_COMPONENT",
    12: "HAS_COMPONENT",
    13: "HAS_SUBCLASS",
    14: "HAS_SUPERCLASS",
    15: "INDUCES",
    16: "INTERACTS_WITH",
    17: "PART_OF",
    18: "PARTICIPATES_IN",
    19: "PARTICIPATES_IN_NEGATIVE_REGULATION_OF",
    20: "PARTICIPATES_IN_POSITIVE_REGULATION_OF",
    21: "PARTICIPATES_IN_REGULATION_OF",
    22: "REGULATES",
    23: "RELATED_TO",
    24: "LOCATED_IN",
    25: "LOCATED_IN_CELLULAR_COMPONENT",
    26: "TARGETS",
    27: "ENCODED_BY",
}

# Group relations by biological category
RELATION_CATEGORIES = {
    "Regulatory": [0, 1, 3, 4, 15, 19, 20, 21, 22],
    "Expression": [5, 10, 27],
    "Structural": [11, 12, 13, 14, 17],
    "Interaction": [2, 16, 26],
    "Location": [24, 25],
    "Process": [6, 7, 8, 9, 18, 23],
}

# Create reverse mapping for category lookup
RELATION_TO_CATEGORY = {}
for category, rel_ids in RELATION_CATEGORIES.items():
    for rel_id in rel_ids:
        RELATION_TO_CATEGORY[rel_id] = category

# Load the dataset
print("Loading the OpenBioLink dataset...")
dl = DataLoader("HQ_DIR")
train = dl.training
test = dl.testing
valid = dl.validation

print("\nOpenBioLink Dataset Biological Exploration")
print("------------------------------------------")

# Print dataset statistics
print(f"\nDataset Statistics:")
print(f"Number of entities: {dl.num_entities}")
print(f"Number of relations: {dl.num_relations}")
print(f"Training triples: {len(train)}")
print(f"Validation triples: {len(valid)}")
print(f"Testing triples: {len(test)}")

# Analyze relation types
print("\nRelation Distribution:")
relation_counts = Counter(train[:, 1].numpy())
relation_df = pd.DataFrame({
    'Relation ID': [rid for rid, _ in relation_counts.most_common()],
    'Name': [RELATION_NAMES.get(rid, f"Unknown({rid})") for rid, _ in relation_counts.most_common()],
    'Category': [RELATION_TO_CATEGORY.get(rid, "Unknown") for rid, _ in relation_counts.most_common()],
    'Count': [count for _, count in relation_counts.most_common()],
    'Percentage': [count/len(train)*100 for _, count in relation_counts.most_common()]
})

# Display the DataFrame
pd.set_option('display.max_rows', None)
print(relation_df)

# Visualize relation distribution by category
plt.figure(figsize=(12, 8))
category_counts = defaultdict(int)
for rel_id, count in relation_counts.items():
    category = RELATION_TO_CATEGORY.get(rel_id, "Unknown")
    category_counts[category] += count

# Create a pie chart
plt.pie(
    [count for category, count in category_counts.items()],
    labels=[f"{category} ({count:,})" for category, count in category_counts.items()],
    autopct='%1.1f%%',
    startangle=90,
    shadow=True,
)
plt.axis('equal')
plt.title('Distribution of Relations by Biological Category', fontsize=16)
plt.tight_layout()
plt.savefig('relation_categories_pie.png')
print("\nSaved relation category distribution pie chart to 'relation_categories_pie.png'")

# Create a bar chart for the top 10 relations
plt.figure(figsize=(14, 8))
top_relations = relation_df.head(10)
colors = sns.color_palette("husl", len(top_relations))
bars = plt.bar(
    top_relations['Name'],
    top_relations['Count'],
    color=[colors[i] for i in range(len(top_relations))],
)
plt.title('Top 10 Relations in OpenBioLink', fontsize=16)
plt.xlabel('Relation Type', fontsize=14)
plt.ylabel('Number of Occurrences', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Add count labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,
        height + 5000,
        f'{int(height):,}',
        ha='center',
        va='bottom',
        rotation=0,
        fontsize=10
    )

plt.savefig('top_relations_bar.png')
print("Saved top relations bar chart to 'top_relations_bar.png'")

# Sample triples for each relation type
print("\nSample Triples by Relation Type:")
for rel_id, name in RELATION_NAMES.items():
    rel_triples = train[train[:, 1] == rel_id]
    if len(rel_triples) > 0:
        print(f"\nSamples for Relation: {name} (ID: {rel_id}, Category: {RELATION_TO_CATEGORY.get(rel_id, 'Unknown')})")
        print(f"Total occurrences: {len(rel_triples)}")
        for i in range(min(3, len(rel_triples))):
            h, r, t = rel_triples[i].tolist()
            print(f"  Entity {h} --[{name}]--> Entity {t}")

# Find the most connected entities
print("\nTop Connected Entities Analysis:")

# Count connections for each entity
entity_connections = defaultdict(int)
for h, _, t in train:
    entity_connections[h.item()] += 1
    entity_connections[t.item()] += 1

# Find the top 5 most connected entities
top_entities = sorted(entity_connections.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 most connected entities:")
for entity_id, connection_count in top_entities:
    print(f"Entity {entity_id}: {connection_count} connections")

    # Get sample connections for this entity
    entity_as_head = train[train[:, 0] == entity_id]
    entity_as_tail = train[train[:, 2] == entity_id]
    
    print(f"  Appears as head in {len(entity_as_head)} triples")
    print(f"  Appears as tail in {len(entity_as_tail)} triples")
    
    if len(entity_as_head) > 0:
        print("  Sample outgoing connections:")
        for i in range(min(3, len(entity_as_head))):
            h, r, t = entity_as_head[i].tolist()
            print(f"    Entity {h} --[{RELATION_NAMES.get(r, f'Relation {r}')}]--> Entity {t}")
    
    if len(entity_as_tail) > 0:
        print("  Sample incoming connections:")
        for i in range(min(3, len(entity_as_tail))):
            h, r, t = entity_as_tail[i].tolist()
            print(f"    Entity {h} --[{RELATION_NAMES.get(r, f'Relation {r}')}]--> Entity {t}")

# Analyze relation pairs and paths
print("\nRelation Path Analysis:")

# Sample a small subgraph to analyze paths (using the most connected entity)
top_entity = top_entities[0][0]
print(f"Analyzing paths from highly connected entity {top_entity}")

# Build a small network around this entity
G = nx.DiGraph()
hop1_triples = train[(train[:, 0] == top_entity) | (train[:, 2] == top_entity)]
hop1_entities = set()

# Add first-hop connections
for h, r, t in hop1_triples:
    h, r, t = h.item(), r.item(), t.item()
    G.add_edge(h, t, relation=RELATION_NAMES.get(r, f"Relation {r}"))
    if h != top_entity:
        hop1_entities.add(h)
    if t != top_entity:
        hop1_entities.add(t)

# Select a few entities from hop1 to explore further
sample_hop1 = list(islice(hop1_entities, 3))
for entity in sample_hop1:
    hop2_triples = train[(train[:, 0] == entity) | (train[:, 2] == entity)]
    for h, r, t in hop2_triples[:5]:  # Limit to prevent explosion
        h, r, t = h.item(), r.item(), t.item()
        G.add_edge(h, t, relation=RELATION_NAMES.get(r, f"Relation {r}"))

print(f"Created a sample subgraph with {G.number_of_nodes()} entities and {G.number_of_edges()} connections")

# Find paths between entities
if len(sample_hop1) >= 2:
    source = sample_hop1[0]
    target = sample_hop1[1]
    
    print(f"\nFinding paths between Entity {source} and Entity {target}:")
    try:
        paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=3))
        if paths:
            print(f"Found {len(paths)} paths (limited to length 3)")
            for i, path in enumerate(paths[:3]):  # Show max 3 paths
                print(f"Path {i+1}: ", end="")
                for j in range(len(path)-1):
                    rel = G.get_edge_data(path[j], path[j+1])['relation']
                    print(f"Entity {path[j]} --[{rel}]--> ", end="")
                print(f"Entity {path[-1]}")
        else:
            print(f"No paths found between Entity {source} and Entity {target} within 3 hops")
    except nx.NetworkXNoPath:
        print(f"No path exists between Entity {source} and Entity {target}")

# Relation co-occurrence analysis
print("\nRelation Co-occurrence Analysis:")
relation_pairs = defaultdict(int)

# Sample a subset of entities to analyze co-occurrence
sampled_entities = np.random.choice(dl.num_entities, size=1000, replace=False)
for entity in sampled_entities:
    # Get all relations involving this entity
    entity_triples = train[(train[:, 0] == entity) | (train[:, 2] == entity)]
    relations = set()
    
    for h, r, t in entity_triples:
        relations.add(r.item())
    
    # Count all relation pairs
    relations = list(relations)
    for i in range(len(relations)):
        for j in range(i+1, len(relations)):
            pair = tuple(sorted([relations[i], relations[j]]))
            relation_pairs[pair] += 1

# Display top co-occurring relation pairs
print("\nTop 10 co-occurring relation pairs:")
top_pairs = sorted(relation_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
for (rel1, rel2), count in top_pairs:
    print(f"{RELATION_NAMES.get(rel1, f'Relation {rel1}')} and {RELATION_NAMES.get(rel2, f'Relation {rel2}')} co-occur {count} times")

print("\nExploration complete!")