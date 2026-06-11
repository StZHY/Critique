# RISC: Recommender with Interactive Critiquing

RISC is a knowledge graph-based recommendation framework that enhances the **KGIN** (Knowledge Graph-based Disentangled Recommendation) model with a **critiquing** mechanism. After a base recommender is trained, RISC simulates user critiquing feedback (keyphrase-level preferences) and fine-tunes user embeddings through importance sampling, improving recommendation quality without requiring explicit user feedback.

## Project Structure

```
RISC/
├── main.py                  # Train the base KGIN recommender
├── simulate.py              # Simulate critiquing and fine-tune the model
├── generate_cri.py          # Generate critique keyphrases for test users
├── modules/
│   ├── KGIN.py              # KGIN model (Aggregator + GraphConv + Recommender)
│   └── RISC.py              # Critique model for critiquing-based refinement
├── utils/
│   ├── parser.py            # Command-line argument parser
│   ├── data_loader.py       # Data loading for base KGIN training
│   ├── cri_data_loader.py   # Data loading for critiquing (with KG mappings)
│   ├── evaluate.py          # Evaluation metrics for KGIN
│   ├── cri_evaluate.py      # Evaluation metrics for the critique model
│   ├── metrics.py           # Metric functions (Recall, NDCG, Precision, Hit, AUC)
│   └── helper.py            # Utility functions (early stopping, file I/O)
```

## Workflow

1. **Base Training** (`main.py`): Train a KGIN recommender on user-item interactions + knowledge graph. Save the model weights.
2. **Critiquing Simulation** (`simulate.py`): Load the pre-trained KGIN, simulate user critique keyphrases via `generate_cri.py`, perform importance sampling to construct training pairs, and fine-tune user embeddings with the Critique model.

## Model Details

### KGIN (`modules/KGIN.py`)

- **Aggregator**: Relational path-aware convolution that aggregates KG neighbor embeddings and user-item interaction signals.
- **GraphConv**: Multi-hop graph convolution with disentangled latent factors and independence regularization (MI / distance correlation / cosine similarity).
- **Recommender**: Wraps GCN with BPR loss, L2 regularization, and disentangle correlation loss.

### Critique (`modules/RISC.py`)

- Loads pre-trained user and entity embeddings from KGIN.
- Appends a weighted-average item embedding per user (Jaccard-based replay probability) as replay positive samples.
- Fine-tunes user embeddings via BPR: push users toward positive items and away from critique keyphrases.

## Quick Start

```bash
# Step 1: Train the base KGIN model
python main.py --dataset last-fm --epoch 1000 --dim 64

# Step 2: Run critiquing simulation
python simulate.py --dataset last-fm --imp_sample gat --cri_epoch 10
```

## Dependencies

- Python 3.8+
- PyTorch
- torch_scatter
- networkx
- scipy
- numpy
- scikit-learn
- prettytable
- tqdm

## Data Format

The dataset directory (e.g., `../data/last-fm/`) should contain:

- `train.txt` — User-item interactions for training. Each line: `user_id item1 item2 ...`
- `test.txt` — User-item interactions for testing. Same format.
- `kg_final.txt` — Knowledge graph triplets. Each line: `head_id relation_id tail_id`
