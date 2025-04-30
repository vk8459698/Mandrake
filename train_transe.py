import torch
import torch.nn as nn
import torch.optim as optim
from openbiolink.evaluation.dataLoader import DataLoader
import numpy as np
from tqdm import tqdm
import os

# Create directory for model results
os.makedirs("model_results", exist_ok=True)

# Define TransE model
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # Normalize entity embeddings
        self.entity_embeddings.weight.data = nn.functional.normalize(
            self.entity_embeddings.weight.data, p=2, dim=1
        )
    
    def forward(self, heads, relations, tails):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        
        scores = head_embeddings + relation_embeddings - tail_embeddings
        return torch.norm(scores, p=2, dim=1)
    
    def predict_tail_scores(self, head, relation, batch_size=1000):
        """
        Memory-efficient prediction of tail scores by processing in batches
        """
        device = head.device
        num_entities = self.entity_embeddings.weight.shape[0]
        scores = torch.zeros(num_entities, device=device)
        
        # Get embeddings for head and relation
        head_emb = self.entity_embeddings(head).unsqueeze(0)  # [1, dim]
        rel_emb = self.relation_embeddings(relation).unsqueeze(0)  # [1, dim]
        
        # Process entity candidates in batches
        for start_idx in range(0, num_entities, batch_size):
            end_idx = min(start_idx + batch_size, num_entities)
            entity_indices = torch.arange(start_idx, end_idx, device=device)
            entity_embs = self.entity_embeddings(entity_indices)  # [batch_size, dim]
            
            # Calculate scores
            batch_scores = head_emb + rel_emb - entity_embs.unsqueeze(0)  # [1, batch_size, dim]
            batch_scores = torch.norm(batch_scores, p=2, dim=2).squeeze(0)  # [batch_size]
            
            scores[start_idx:end_idx] = batch_scores
            
        return scores

# Load data
print("Loading data...")
dl = DataLoader("HQ_DIR")
train_triples = dl.training
valid_triples = dl.validation
test_triples = dl.testing

num_entities = dl.num_entities
num_relations = dl.num_relations

print(f"Number of entities: {num_entities}")
print(f"Number of relations: {num_relations}")
print(f"Training triples: {len(train_triples)}")
print(f"Validation triples: {len(valid_triples)}")
print(f"Test triples: {len(test_triples)}")

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = TransE(num_entities, num_relations, embedding_dim=100).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
margin = 1.0

# Generate negative samples
def generate_negatives(positive_triples, batch_size, num_entities):
    # Copy the batch
    negative_triples = positive_triples.clone()
    
    # Corrupt either head or tail
    for i in range(batch_size):
        if np.random.random() < 0.5:
            # Corrupt head
            negative_triples[i, 0] = np.random.randint(0, num_entities)
        else:
            # Corrupt tail
            negative_triples[i, 2] = np.random.randint(0, num_entities)
    
    return negative_triples

# Training loop
def train(model, train_triples, optimizer, batch_size=512, epochs=5):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        indices = np.random.permutation(len(train_triples))
        
        for start_idx in tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch = train_triples[batch_indices].to(device)
            
            # Generate negative samples
            negatives = generate_negatives(batch, len(batch_indices), num_entities).to(device)
            
            # Forward pass
            pos_scores = model(batch[:, 0], batch[:, 1], batch[:, 2])
            neg_scores = model(negatives[:, 0], negatives[:, 1], negatives[:, 2])
            
            # Calculate loss
            loss = torch.mean(torch.relu(margin + pos_scores - neg_scores))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize entity embeddings
            with torch.no_grad():
                model.entity_embeddings.weight.data = nn.functional.normalize(
                    model.entity_embeddings.weight.data, p=2, dim=1
                )
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(indices) / batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model

# Memory-efficient evaluation
def evaluate(model, test_triples, k=10, batch_size=64):
    model.eval()
    hits_at_k = 0
    mean_rank = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_triples), batch_size), desc="Evaluating"):
            batch = test_triples[i:i+batch_size].to(device)
            
            batch_hits = 0
            batch_ranks = 0
            
            for j in range(len(batch)):
                h, r, t = batch[j]
                
                # Get scores for all possible tails
                tail_scores = model.predict_tail_scores(h, r)
                
                # Get the score of the true tail
                true_score = tail_scores[t].item()
                
                # Count entities with better scores than the true tail
                rank = 1 + (tail_scores < true_score).sum().item()
                
                if rank <= k:
                    batch_hits += 1
                batch_ranks += rank
            
            hits_at_k += batch_hits
            mean_rank += batch_ranks
    
    hits_at_k /= len(test_triples)
    mean_rank /= len(test_triples)
    
    return hits_at_k, mean_rank

# Train model (using a small subset for demo purposes)
print("Training model...")
# Use a small subset for quick demonstration
train_subset_size = min(200000, len(train_triples))
train_subset = train_triples[:train_subset_size]
model = train(model, train_subset, optimizer, batch_size=512, epochs=3)

# Save the model
torch.save(model.state_dict(), "model_results/transe_model.pt")

# Evaluate on a small subset
print("Evaluating model...")
test_subset_size = min(5000, len(test_triples))
test_subset = test_triples[:test_subset_size]
hits_at_10, mean_rank = evaluate(model, test_subset, k=10, batch_size=64)

print(f"Hits@10: {hits_at_10:.4f}")
print(f"Mean Rank: {mean_rank:.2f}")

print("Model training and evaluation complete!")