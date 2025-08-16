# DF-EL++
This repository provides the code, data, and experimental details for the KDD 2026 paper: "Fast and Faithful: Scalable Neuro-Symbolic Learning and Reasoning with Differentiable Fuzzy EL++".

## Overview

![项目概览图](./images/workflow1.png)

This work introduces DF-EL++, the first end-to-end differentiable framework that unifies PTIME-complete reasoning with neural learning, resolving the persistent trade-off between logical rigor and computational scale in neuro-symbolic AI.

Our framework operates as a principled refinement engine: a neural network first grounds symbols by producing an initial, uncertain fuzzy knowledge base from data. DF-EL++ then uses gradient-based optimization to repair and refine these beliefs, ensuring they cohere with an ontology's logical constraints while preserving high-confidence empirical evidence. This creates a virtuous cycle where perception is disciplined by logic, and logic is grounded in data .

Validated on massive, real-world ontologies like SNOMED CT (377K concepts), DF-EL++ demonstrates a unique synergy of scale and performance: it remains computationally efficient where expressive systems fail, while decisively outperforming dominant scalable baselines in a range of knowledge base completion tasks with up to a 42% relative improvement in Hits@1.

## Details for Reproduction

## Dependencies
All experiments were conducted on servers equipped with NVIDIA RTX 3090 GPUs. The framework is implemented in PyTorch and requires the following dependencies:
    
    JDK 1.8
    Python: 3.7.0+
    PyTorch: 1.12
    CUDA: 11.6
    NumPy: 1.21.4
    Pandas: 1.1.3
    Matplotlib: 3.3.2
    python-csv: 0.0.13
    pickle: 4.0
    pyparsing: 3.0.6
    loguru: 0.6.0

### Normalization of EL++ ontologies

The input EL++ ontology must be normalized before training. This process does not truncate the logic but instead systematically decomposes complex axioms into a set of elementary "normal forms" (NF1-NF6). Each normal form contains at most one logical operator beyond subsumption, which is a necessary precondition for stable and effective gradient-based learning. This transformation preserves the original semantics of the ontology.

To run the normalization on your ontology, use the following script with **JDK 1.8**:
    
    java -jar Normalization.jar training/ontologies training/input

The output of preprocessing is the files in 'training/input':

- 'concepts.txt', 'roles.txt', 'individuals.txt': the concept names(/role names/individual names) set.
- 'normalization.txt': the nomalized TBox axioms.
- 'abox.txt': the abox assertions.
- 'subclassaixoms.txt': the original TBox axioms.

Note: The source code of 'Normalization.jar' and 'CQGenerator.jar' is in [normalization](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/normalization). If you want to repackage the jar based on our source code, remember to delete all dependencies named 'owlapi-xxx.jar' in the artifact, while only remain the 'owlapi-distribution-5.1.3.jar'.

## Training
### Knowledge Base Completion (KBC) Training
The KBC task evaluates a model's ability to predict missing axioms in an ontology. All models are trained using the Adam optimizer with early stopping based on validation MRR (patience: 10 epochs).

### Training DF-EL++ on SNOMED CT
    
    python train_kbc.py \
    --dataset SNOMED_CT \
    --model_name DF-EL++ \
    --learning_rate 2e-4 \
    --embedding_dim 200 \
    --batch_size 512 \
    --l2_lambda 1e-5 \
    --negative_samples 50

### Training Baseline Models
The same script is used for training baselines by changing the --model_name parameter.
### Example: Training Box2EL on Gene Ontology (GO)
    python train_kbc.py \
    --dataset GO \
    --model_name Box2EL \
    --learning_rate 2e-4 \
    --embedding_dim 200 \
    --batch_size 1024 \
    --l2_lambda 1e-5 \
    --negative_samples 50

### Semantic Image Interpretation (SII) Training
The SII task evaluates the framework's ability to refine noisy outputs from a pre-trained perception model (FR-CNN) to be consistent with a domain ontology.
### Training DF-EL++ for SII
    python train_sii.py \
    --dataset PASCAL_PART \
    --model_name DF-EL++ \
    --learning_rate 1e-4 \
    --max_epochs 50
### Additional Training Examples
for dataset in SNOMED_CT GO Yeast_PPI Human_PPI; do
    
    python train_kbc.py \
    --dataset $dataset \
    --model_name DF-EL++ \
    --learning_rate 2e-4 \
    --embedding_dim 200 \
    --batch_size $([ "$dataset" = "SNOMED_CT" ] && echo 512 || echo 1024) \
    --l2_lambda 1e-5 \
    --negative_samples 50

done

# Train on all SII datasets
for dataset in PASCAL_PART ADE20K PartImageNet; do
    
    python train_sii.py \
    --dataset $dataset \
    --model_name DF-EL++ \
    --learning_rate 1e-4 \
    --max_epochs 50

done

#### KBC Task Settings

| Parameter | Search Space | Optimal (DF-EL++) |
|-----------|-------------|-------------------|
| Optimizer | - | Adam |
| Learning Rate | {1e-4, 2e-4, 5e-4} | 2e-4 |
| Embedding Dimension | {100, 200, 400} | 200 |
| Batch Size | {256, 512, 1024} | 512 (SNOMED CT) / 1024 (others) |
| L2 Regularization (λ) | - | 1e-5 |
| Negative Samples | - | 50 |
| Margin (δ) | - | 1.0 |
| Loss Function | - | Self-adversarial negative sampling |
| Early Stopping | - | 10 epochs (patience on val MRR) |

#### SII Task Settings

| Parameter | Configuration | Notes |
|-----------|---------------|-------|
| Optimizer | Adam | For refinement stage |
| Learning Rate | 1e-4 | Selected from {1e-5, 1e-4, 5e-4} |
| Max Epochs | 50 | Per image refinement |
| Perception Model | Fast R-CNN (ResNet-50) | Pre-trained backbone |
| Input | FR-CNN detections | Initial fuzzy beliefs |
| Task Type | Fuzzy belief optimization | Logical consistency refinement |


# DF-EL++

This repository provides the code, data, and experimental details for the KDD 2026 paper: "Fast and Faithful: Scalable Neuro-Symbolic Learning and Reasoning with Differentiable Fuzzy EL++".

## Overview

This work introduces DF-EL++, the first end-to-end differentiable framework that unifies PTIME-complete reasoning with neural learning, resolving the persistent trade-off between logical rigor and computational scale in neuro-symbolic AI.

Our framework operates as a principled refinement engine: a neural network first grounds symbols by producing an initial, uncertain fuzzy knowledge base from data. DF-EL++ then uses gradient-based optimization to repair and refine these beliefs, ensuring they cohere with an ontology's logical constraints while preserving high-confidence empirical evidence. This creates a virtuous cycle where perception is disciplined by logic, and logic is grounded in data.

Validated on massive, real-world ontologies like SNOMED CT (377K concepts), DF-EL++ demonstrates a unique synergy of scale and performance: it remains computationally efficient where expressive systems fail, while decisively outperforming dominant scalable baselines in a range of knowledge base completion tasks with up to a 42% relative improvement in Hits@1.

## Details for Reproduction

## Dependencies

All experiments were conducted on servers equipped with NVIDIA RTX 3090 GPUs. The framework is implemented in PyTorch and requires the following dependencies:

- JDK 1.8
- Python: 3.7.0+
- PyTorch: 1.12
- CUDA: 11.6
- NumPy: 1.21.4
- Pandas: 1.1.3
- Matplotlib: 3.3.2
- python-csv: 0.0.13
- pickle: 4.0
- pyparsing: 3.0.6
- loguru: 0.6.0

### Normalization of EL++ ontologies

The input EL++ ontology must be normalized before training. This process does not truncate the logic but instead systematically decomposes complex axioms into a set of elementary "normal forms" (NF1-NF6). Each normal form contains at most one logical operator beyond subsumption, which is a necessary precondition for stable and effective gradient-based learning. This transformation preserves the original semantics of the ontology.

To run the normalization on your ontology, use the following script with **JDK 1.8**:

```bash
java -jar Normalization.jar training/ontologies training/input
```

The output of preprocessing is the files in `training/input`:

- `concepts.txt`, `roles.txt`, `individuals.txt`: the concept names(/role names/individual names) set.
- `normalization.txt`: the nomalized TBox axioms.
- `abox.txt`: the abox assertions.
- `subclassaixoms.txt`: the original TBox axioms.

**Note:** The source code of `Normalization.jar` and `CQGenerator.jar` is in [normalization](https://github.com/AnonymousResearcherOpen/NeSyALC/tree/main/normalization). If you want to repackage the jar based on our source code, remember to delete all dependencies named `owlapi-xxx.jar` in the artifact, while only remain the `owlapi-distribution-5.1.3.jar`.

## Training

### Knowledge Base Completion (KBC) Training

The KBC task evaluates a model's ability to predict missing axioms in an ontology. All models are trained using the Adam optimizer with early stopping based on validation MRR (patience: 10 epochs).

#### Training DF-EL++ on SNOMED CT

```bash
python train_kbc.py \
  --dataset SNOMED_CT \
  --model_name DF-EL++ \
  --learning_rate 2e-4 \
  --embedding_dim 200 \
  --batch_size 512 \
  --l2_lambda 1e-5 \
  --negative_samples 50
```

### Training Baseline Models

The same script is used for training baselines by changing the `--model_name` parameter.

#### Example: Training Box2EL on Gene Ontology (GO)

```bash
python train_kbc.py \
  --dataset GO \
  --model_name Box2EL \
  --learning_rate 2e-4 \
  --embedding_dim 200 \
  --batch_size 1024 \
  --l2_lambda 1e-5 \
  --negative_samples 50
```

### Semantic Image Interpretation (SII) Training

The SII task evaluates the framework's ability to refine noisy outputs from a pre-trained perception model (FR-CNN) to be consistent with a domain ontology.

#### Training DF-EL++ for SII

```bash
python train_sii.py \
  --dataset PASCAL_PART \
  --model_name DF-EL++ \
  --learning_rate 1e-4 \
  --max_epochs 50
```

### Additional Training Examples

```bash
# Train on all KBC datasets
for dataset in SNOMED_CT GO Yeast_PPI Human_PPI; do
  python train_kbc.py \
    --dataset $dataset \
    --model_name DF-EL++ \
    --learning_rate 2e-4 \
    --embedding_dim 200 \
    --batch_size $([ "$dataset" = "SNOMED_CT" ] && echo 512 || echo 1024) \
    --l2_lambda 1e-5 \
    --negative_samples 50
done

# Train on all SII datasets
for dataset in PASCAL_PART ADE20K PartImageNet; do
  python train_sii.py \
    --dataset $dataset \
    --model_name DF-EL++ \
    --learning_rate 1e-4 \
    --max_epochs 50
done
```

## Hyperparameter Settings

### KBC Task Settings

| Parameter | Search Space | Optimal (DF-EL++) |
|-----------|-------------|-------------------|
| Optimizer | - | Adam |
| Learning Rate | {1e-4, 2e-4, 5e-4} | 2e-4 |
| Embedding Dimension | {100, 200, 400} | 200 |
| Batch Size | {256, 512, 1024} | 512 (SNOMED CT) / 1024 (others) |
| L2 Regularization (λ) | - | 1e-5 |
| Negative Samples | - | 50 |
| Margin (δ) | - | 1.0 |
| Loss Function | - | Self-adversarial negative sampling |
| Early Stopping | - | 10 epochs (patience on val MRR) |

### SII Task Settings

| Parameter | Configuration | Notes |
|-----------|---------------|-------|
| Optimizer | Adam | For refinement stage |
| Learning Rate | 1e-4 | Selected from {1e-5, 1e-4, 5e-4} |
| Max Epochs | 50 | Per image refinement |
| Perception Model | Fast R-CNN (ResNet-50) | Pre-trained backbone |
| Input | FR-CNN detections | Initial fuzzy beliefs |
| Task Type | Fuzzy belief optimization | Logical consistency refinement |
