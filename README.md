# GUARDNET 🧠

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.6%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Learning to Reason over Neighborhoods: A Differentiable Guarded Logic Approach**  
> *Under review at ICLR 2026*

This repository provides the official PyTorch implementation for **GUARDNET**. The code is designed to be a complete and faithful reproduction of the paper's methodology, enabling researchers to explore and build upon our work in scalable neuro-symbolic reasoning.

---

## 📋 Overview

**GUARDNET** is the first framework to leverage the **Guarded Fragment (GF)** of first-order logic as a principled inductive bias for robust and scalable neighborhood-based reasoning. It directly addresses the critical challenge of systematic generalization in deep learning, particularly in tasks requiring multi-hop inference, where existing models often fail.

### 🎯 Key Features

- **Guarded Logic as Inductive Bias**: Employs the syntactic 'guard' of GF to restrict logical quantification to local, relational neighborhoods
- **Hybrid Domain Training**: Novel training strategy combining **Core Domain** with **Latent Domain** for logical fidelity and robust generalization
- **Systematic Generalization**: Exceptional performance on challenging zero-shot, multi-hop reasoning tasks

---

## 🛠️ Installation & Setup

### Prerequisites

The framework is built entirely in Python. All experiments were conducted on servers equipped with NVIDIA GPUs.

| Component | Version | Purpose |
|:----------|:--------|:--------|
| **Hardware** | NVIDIA RTX 4090 (24GB) | Recommended for GPU acceleration |
| **Python** | 3.7.0+ | Core framework language |
| **PyTorch** | 1.12.1+ | Deep learning backend |
| **CUDA** | 11.6+ | For GPU support |

### Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/guardnet.git
cd guardnet

# Install Python dependencies
pip install -r requirements.txt
```

<details>
<summary>📦 Full Dependency List</summary>

```
torch>=1.12.1
numpy>=1.21.0
pandas>=1.1.3
matplotlib>=3.3.2
scikit-learn>=0.24.0
tqdm>=4.62.0
loguru>=0.6.0
pyparsing>=3.0.6
```

</details>

---

## 🔧 Data Preparation

The framework expects standard knowledge base completion datasets, split into training, validation, and test sets. 

The datasets referenced in the paper include:

- SNOMED CT
- Gene Ontology (GO)
- Yeast PPI
- Human PPI
- FB15k-237
- WN18RR

### 📁 Required Data Structure

Please place your datasets inside a `data/` directory, following this structure for each dataset:
```
data/your_dataset_name/
├── entities.txt      # List of all entity names, one per line
├── relations.txt     # List of all relation (predicate) names
├── train.txt         # Training facts (head_entity relation tail_entity)
├── valid.txt         # Validation facts
└── test.txt          # Test facts
```

The axioms (rules) for training should be provided in a separate file or defined within the data loading logic, following the syntax described in the paper.

---

## 🚀 Training & Evaluation

The code has been structured into separate files for clarity. The main entry point is `main.py`.

### 🧪 Zero-Shot Multi-Hop Generalization

This is a special evaluation task designed to rigorously test the model's ability to generalize to longer reasoning chains not seen during training.

To run this evaluation, first generate the special data splits where the training set contains only 1- and 2-hop provable facts, and the test set contains only 3+ hop facts.
```bash
# Generate the multi-hop data splits
python generate_multihop_splits.py \
    --data_path data/your_dataset_name \
    --output_path data/your_dataset_name_multihop

# Train the model on the 1-2 hop training data
python main.py

# Evaluate the trained model on the 3+ hop test data
python evaluate.py \
    --model_path checkpoints/best_model.pt \
    --data_path data/your_dataset_name_multihop
```

---

## ⚙️ Hyperparameter Configuration

The optimal hyperparameters found in the paper are set as defaults in the `config.py` file.

| Parameter | Selected Value | Description |
|:----------|:--------------|:------------|
| **Optimizer** | AdamW | Decoupled weight decay optimizer |
| **Learning Rate** | 5e-4 | Initial learning rate |
| **Embedding Dim** | 200 | Dimensionality of constant embeddings |
| **Batch Size** | 512 | Samples per iteration |
| **Weight Decay** | 5e-5 | L2 regularization coefficient |
| **Predicate MLP** | 2 hidden layers, 256 units, ReLU | Architecture for grounding predicates |
| **LSE Temperature** | 0.1 | Controls sharpness of quantifier approximation |
| **Loss Curriculum** | 0.9 to 0.4 | Linearly anneals from fidelity to generalization |
| **Early Stopping** | 15 epochs patience | Prevents overfitting |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ for the Neuro-Symbolic AI Community</sub>
</div>
