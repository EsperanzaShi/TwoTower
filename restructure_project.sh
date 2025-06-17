#!/bin/bash

# Exit if any command fails
set -e

echo "📁 Creating new directory structure..."

mkdir -p src/two_tower
mkdir -p scripts
mkdir -p data
mkdir -p figures

# Move core modules
mv model.py src/two_tower/
mv optimizer.py src/two_tower/
touch src/two_tower/__init__.py

# Move scripts
mv train.py scripts/
mv generate_triplets.py scripts/
mv tokenize_dataset.py scripts/
mv inspect_triplets.py scripts/
mv visualize_tsne.py scripts/
mv visualize_single_triplet.py scripts/
mv wandb_Artifact.py scripts/

# Move data
mv .data/*.pkl data/
rmdir .data || true  # remove only if empty

# Move figures
mv tsne_after_training1epoch.png figures/ 2>/dev/null || true

echo "✅ Project restructured!"