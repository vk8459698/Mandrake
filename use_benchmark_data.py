from openbiolink.evaluation.dataLoader import DataLoader
import os

# Create a directory to store downloaded data
os.makedirs("benchmark_data", exist_ok=True)
os.chdir("benchmark_data")

print("Loading benchmark dataset...")
# Load the high-quality directed dataset (default)
dl = DataLoader("HQ_DIR")

# The data is directly available in the DataLoader object
train = dl.training
test = dl.testing
valid = dl.validation

print(f"Training samples: {len(train)}")
print(f"Testing samples: {len(test)}")
print(f"Validation samples: {len(valid)}")

# Display some training examples
print("\nSample training examples:")
for i in range(min(5, len(train))):
    print(train[i])

# Get information about entity and relation IDs
print("\nExploring dataset structure:")
print("First, let's check what attributes are available in the DataLoader")
for attr in dir(dl):
    if not attr.startswith('_'):
        print(f"- {attr}")

# Try to access entity and relation information if available
try:
    if hasattr(dl, 'dataset'):
        print("\nDataset attributes:")
        for attr in dir(dl.dataset):
            if not attr.startswith('_'):
                print(f"- {attr}")
except Exception as e:
    print(f"Error accessing dataset attributes: {e}")

print("\nBenchmark dataset loaded successfully!")

# Let's print the shape of the tensors to understand the data structure
print("\nData shapes:")
print(f"Training tensor shape: {train.shape}")
print(f"Testing tensor shape: {test.shape}")
print(f"Validation tensor shape: {valid.shape}")
