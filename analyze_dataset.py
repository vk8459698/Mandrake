from openbiolink.evaluation.dataLoader import DataLoader
import torch
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt

# Create directory for analysis results
os.makedirs("analysis_results", exist_ok=True)

# Load the dataset
dl = DataLoader("HQ_DIR")
train = dl.training
test = dl.testing
valid = dl.validation

print(f"Number of entities: {dl.num_entities}")
print(f"Number of relations: {dl.num_relations}")

# Analyze relation distribution
print("\nAnalyzing relation distribution...")
relation_counts = Counter(train[:, 1].numpy())
print(f"Number of unique relations in training: {len(relation_counts)}")

# Get top 5 most common relations
top_relations = relation_counts.most_common(5)
print("\nTop 5 most common relations:")
for rel_id, count in top_relations:
    print(f"Relation ID {rel_id}: {count} occurrences ({count/len(train)*100:.2f}%)")

# Create a histogram of relation frequency
plt.figure(figsize=(12, 6))
plt.bar(range(len(relation_counts)), [count for _, count in relation_counts.most_common()])
plt.xlabel('Relation Index (sorted by frequency)')
plt.ylabel('Frequency')
plt.title('Relation Frequency Distribution')
plt.savefig('analysis_results/relation_distribution.png')

# Analyze connectivity
print("\nAnalyzing entity connectivity...")
head_entities = Counter(train[:, 0].numpy())
tail_entities = Counter(train[:, 2].numpy())

print(f"Number of unique head entities: {len(head_entities)}")
print(f"Number of unique tail entities: {len(tail_entities)}")

# Compute statistics on connectivity
head_degrees = list(head_entities.values())
tail_degrees = list(tail_entities.values())

print(f"Average outgoing connections per entity: {np.mean(head_degrees):.2f}")
print(f"Average incoming connections per entity: {np.mean(tail_degrees):.2f}")
print(f"Max outgoing connections: {max(head_degrees)}")
print(f"Max incoming connections: {max(tail_degrees)}")

# Create degree distribution plots
plt.figure(figsize=(12, 6))
plt.hist(np.log10(head_degrees), bins=50, alpha=0.7, label='Outgoing connections')
plt.hist(np.log10(tail_degrees), bins=50, alpha=0.7, label='Incoming connections')
plt.xlabel('Log10(Degree)')
plt.ylabel('Count')
plt.title('Entity Connection Distribution')
plt.legend()
plt.savefig('analysis_results/degree_distribution.png')

print("\nAnalysis complete. Results saved in 'analysis_results' directory.")
