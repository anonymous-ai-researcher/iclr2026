\# Fast and Faithful: A Scalable Neuro-Symbolic Framework for Ontological Reasoning



\[!\[Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch\&logoColor=white)](https://pytorch.org/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

\[!\[KDD 2026](https://img.shields.io/badge/KDD-2026-blue)](https://kdd.org/kdd2026/)



This repository contains the official PyTorch implementation for the KDD 2026 paper: \*\*"Fast and Faithful: Scalable Neuro-Symbolic Learning and Reasoning with Differentiable Fuzzy EL++"\*\*.



Our work introduces \*\*DF-EL++\*\*, the first end-to-end differentiable framework that resolves the persistent trade-off between logical rigor and computational scale in neuro-symbolic AI. By unifying the PTIME-complete reasoning of the description logic $\\mathcal{EL}^{++}$ with a principled, Product-based fuzzy semantics, DF-EL++ enables scalable and logically sound reasoning on massive, real-world knowledge bases where other methods fail.



\## Framework Overview



Our framework operates as a principled refinement engine. A neural network first grounds symbols by producing an initial, uncertain fuzzy knowledge base from data. DF-EL++ then uses gradient-based optimization to repair and refine these beliefs, ensuring they cohere with the ontology's logical constraints.



!\[Framework Overview](https://i.imgur.com/your\_diagram\_link.png)

\*Figure 1: The DF-EL++ framework, illustrating the full pipeline from neural perception to logically consistent belief refinement.\*



\## Key Contributions



\* \*\*A Scalable and Differentiable Framework for EL++:\*\* We introduce the first end-to-end differentiable learning framework built upon the description logic $\\mathcal{EL}^{++}$, enabling scalable neuro-symbolic inference on massive, real-world ontologies.

\* \*\*A Complete, Theoretically-Grounded Methodology:\*\* We propose a complete methodology for differentiable fuzzy reasoning, including a principled selection of Product-based fuzzy semantics, a novel semantically-aware domain construction strategy to prevent model collapse, and a loss function derived directly from the corresponding Göguen R-implication.

\* \*\*Extensive Empirical Validation:\*\* We conduct a comprehensive evaluation on challenging biomedical ontologies, demonstrating that DF-EL++ is not only computationally efficient but also significantly outperforms state-of-the-art geometric embedding methods in knowledge base completion tasks.



\## Repository Structure

├── data/                  # Scripts for downloading and preprocessing datasets

│   ├── GO/

│   ├── SNOMED\_CT/

│   └── ...

├── src/                   # Source code for the DF-EL++ framework

│   ├── models/            # Model implementations (DF-EL++, baselines)

│   ├── dataloaders.py

│   ├── loss.py            # Implementation of the Göguen-based loss

│   └── ...

├── experiments/           # Scripts to reproduce the paper's experiments

│   ├── run\_kbc.sh

│   └── run\_ablation.sh

├── requirements.txt       # Python dependencies

└── README.md



\## Installation



To set up the environment, we recommend using `conda`.



1\. \*\*Clone the repository:\*\*

&nbsp;  ```bash

&nbsp;  git clone https://github.com/anonymous-ai-researcher/kdd2026.git

&nbsp;  cd kdd2026



Create a Conda environment and install dependencies:

conda create -n df-elpp python=3.9

conda activate df-elpp

pip install -r requirements.txt



Usage: Reproducing Experiments

The experiments are managed through shell scripts in the experiments/ directory.



Data Preparation

First, download and preprocess the datasets:

bash data/download\_and\_preprocess.sh



Training a Model

You can train a new DF-EL++ model or any of the baselines using train.py. The script arguments allow you to specify the model, dataset, and key hyperparameters.



Example: Training DF-EL++ on Gene Ontology (GO)



python src/train.py \\

&nbsp;   --model DF-EL++ \\

&nbsp;   --dataset GO \\

&nbsp;   --embedding\_dim 200 \\

&nbsp;   --batch\_size 1024 \\

&nbsp;   --learning\_rate 2e-4 \\

&nbsp;   --l2\_reg 1e-5 \\

&nbsp;   --num\_neg\_samples 50 \\

&nbsp;   --output\_dir checkpoints/df-elpp-go



Evaluation

To evaluate a trained model, use the evaluate.py script, pointing it to the saved model checkpoint



python src/evaluate.py \\

&nbsp;   --checkpoint\_path checkpoints/df-elpp-go/best\_model.pth \\

&nbsp;   --dataset GO

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions about the paper or the code, please contact the authors at yizheng.zhao1@gmail.com.
