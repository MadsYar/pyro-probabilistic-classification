# Republican or Democrat? A Model-Based Machine Learning Approach for Classifying Tweets

## Overview

This project presents a **model-driven machine learning approach** to classify political tweets as either Republican or Democrat. Rather than using off-the-shelf classifiers, we built custom Bayesian probabilistic models using **Pyro**, a probabilistic programming library built on PyTorch. The work demonstrates how probabilistic modeling can capture uncertainty and complex patterns in social media text data.

This repository contains a fully runnable Jupyter notebook that implements the entire pipeline: data preprocessing, feature engineering, model design, training via Stochastic Variational Inference (SVI), evaluation, and visualization.

---

## Project Motivation

Political discourse on social media is complex and nuanced. Understanding and classifying political opinions requires more than simple keyword matching—it requires models that can learn contextual patterns and capture uncertainty. This project was built to explore whether a custom Bayesian hierarchical model could outperform a traditional Bayesian classifier by explicitly modeling interactions between sentiment, topics, and structural tweet features.

---

## Methodology

### 1. Data Preprocessing

We built a comprehensive preprocessing pipeline that transforms raw tweets into features suitable for probabilistic modeling:

- **Text Cleaning**: Removed URLs, mentions, hashtags, emojis, and non-alphabetic characters. All text converted to lowercase for consistency.
- **Labeling Strategy**: Used "frequent names" associated with each candidate (e.g., "Trump", "MAGA", "Republican" for Republican tweets, and "Biden", "Sleepy Joe", "Democrat" for Democrat tweets) to automatically label 1.2 million tweets.
- **Class Balancing**: Randomly sampled 600,000 tweets per class to avoid class imbalance bias during training.
- **Train-Test Split**: 80% training / 20% testing using stratified sampling.

### 2. Feature Engineering

The notebook extracts five complementary feature sets:

- **TF-IDF Features** (1,000 dimensions): Captures word importance while removing political stop words that provide no discriminative signal.
- **LDA Topic Distributions** (10 topics): Latent Dirichlet Allocation models semantic themes in tweets, representing each tweet as a mixture over topics.
- **Sentiment Features** (4 dimensions): VADER sentiment analysis extracts positive, negative, neutral, and compound sentiment scores.
- **Statistical Features** (6 dimensions): Text length, unique word ratio, uppercase ratio, punctuation count, hashtag count, mention count.
- **Topic-Sentiment Interactions** (10 dimensions): Multiplicative interactions between LDA topic distributions and sentiment scores to capture how sentiment varies by topic.

### 3. Models

We implemented and compared two Bayesian models:

#### **Bayesian Logistic Regression (BLR)**
A traditional probabilistic classifier that serves as a baseline:
- Treats weights as random variables with Normal priors: $\mathcal{N}(0, 1)$
- Final prediction uses sigmoid: $P(y=1|X) = \sigma(X \cdot w + b)$
- Trained via Stochastic Variational Inference (SVI)

#### **Bayesian Hierarchical Model (BHM)** — Custom Design
Our original multi-component model that integrates multiple probabilistic sub-models:

- **Gaussian Mixture Model (GMM)**: Models the distribution of topic-sentiment features across K=3 clusters
  - Each tweet assigned to a cluster with mixture weights
  - Cluster means and variances learned from data
  
- **LDA Component**: Weighted contribution of topic distributions
  - Topic weights learned with global scale hyperparameter
  
- **Sentiment-Topic Interactions**: Learns how sentiment impact depends on topic context
  - Separate weights for each sentiment dimension × topic pair
  
- **Statistical Features**: Text properties contribute via learned weights
  
- **Final Prediction**: Combines all components:
  $$\text{logits} = \text{logits}_{gmm} + \text{logits}_{lda} + \text{sentiment\_effect} + \text{logits}_{stats} + \text{logits}_{tfidf}$$

The BHM's key innovation is **cross-feature dependencies**—sentiment impact is modulated by topics, and statistical features are combined hierarchically rather than independently.

### 4. Training

Both models trained via **Stochastic Variational Inference (SVI)** with:
- **Optimizer**: ClippedAdam (Adam with gradient clipping for stability)
- **Loss**: Evidence Lower Bound (ELBO) — standard for variational inference
- **Iterations**: 2,000 per model
- **Learning Rate**: 0.003
- **Early Stopping**: Patience of 100 iterations (stops if validation accuracy doesn't improve)
- **Validation**: 20% of training data held out to monitor generalization

### 5. Evaluation

Models evaluated on 240,000 test tweets using:
- **Accuracy**: Fraction of correct predictions
- **Precision, Recall, F1**: Per-class performance metrics
- **ROC-AUC**: Ability to distinguish between classes across all thresholds
- **Confusion Matrix**: Breakdown of correct vs. incorrect predictions per class

---

## Key Results

| Metric | BLR | BHM |
|--------|-----|-----|
| **Test Accuracy** | 59.9% | 60.8% |
| **Precision** (weighted) | 60.1% | 61.2% |
| **Recall** (weighted) | 59.9% | 60.8% |
| **ROC-AUC** | 0.638 | 0.712 |
| **Confusion Matrix** | Biased toward Democrat | More balanced |

**Key Finding**: While BLR achieved good accuracy on Democrat tweets (80,053/120,000 correct), it struggled with Republican tweets (66,999/120,000 correct). The BHM achieved better balance, correctly classifying more Republican tweets (79,819/120,000) while maintaining strong Democrat performance. The significantly higher AUC (0.712 vs 0.638) confirms the BHM better captures class separability.

---

## Why Probabilistic Modeling?

Traditional machine learning models (logistic regression, random forests) treat parameters as fixed unknowns. Bayesian probabilistic models treat parameters as random variables with uncertainty. This allows us to:

1. **Quantify Uncertainty**: Posterior distributions reflect what we've learned and how confident we are
2. **Incorporate Domain Knowledge**: Priors encode what we believe before seeing data
3. **Model Complex Relationships**: Hierarchical structures capture multi-level dependencies
4. **Avoid Overfitting**: Regularization through priors and uncertainty quantification

Pyro makes this accessible by automating variational inference, letting us focus on model design.

---

## Quick Start

### Prerequisites
- Python 3.12
- GPU recommended (NVIDIA RTX 3070 or better) but CPU works (slower)

### Installation

```bash
# Clone or download this repository
cd mbml_project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

1. Ensure `data/hashtag_donaldtrump.csv` and `data/hashtag_joebiden.csv` are in the `data/` folder
2. Open and run [Group15_final_notebook.ipynb](Group15_final_notebook.ipynb) top-to-bottom

The notebook is self-contained and fully documented. Each section explains what it does and why.

---

## Runtime Expectations

**Total time: ~103 minutes on a modern desktop (i7-13700KF + RTX 3070)**
- Data preprocessing: ~15 minutes
- Model training: ~85 minutes (most time spent here via SVI)
- Evaluation & visualization: ~3 minutes

CPU-only systems will be significantly slower. Consider using a GPU or reducing dataset size for testing.

---

## Project Structure

```
mbml_project/
├── Group15_final_notebook.ipynb       # Main notebook (self-contained)
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── data/
│   ├── hashtag_donaldtrump.csv        # Raw Trump tweets
│   └── hashtag_joebiden.csv           # Raw Biden tweets
├── Model/
│   └── First/                         # Saved model parameters
│       ├── blr_params.pt
│       ├── bhm_params.pt
│       └── naive_bayes_params.pt
└── Saved Results/
    ├── Save 1/                        # Training & evaluation artifacts
    │   ├── Models/
    │   │   ├── blr_params.pt
    │   │   └── bhm_params.pt
    │   └── Results/
    │       ├── Training/
    │       │   ├── blr_model_results.pkl
    │       │   └── bhm_model_results.pkl
    │       └── Evaluation/
    │           ├── blr_eval_results.pkl
    │           └── bhm_eval_results.pkl
    └── Save 2-7/                      # Previous runs
```

---

## Reproducibility

Random seeds are set in the notebook to ensure consistent results across runs:

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
pyro.set_rng_seed(42)
```

The preprocessing pipeline and model architectures are deterministic given these seeds.

---

## Technical Highlights

- **Probabilistic Programming**: Custom Bayesian models with Pyro and PyTorch
- **Variational Inference**: SVI for scalable approximate posterior inference
- **Feature Engineering**: Multiple complementary feature sets (TF-IDF, LDA, sentiment, statistics)
- **Hierarchical Modeling**: Cross-feature dependencies and multi-component likelihood
- **Comprehensive Evaluation**: Metrics, confusion matrices, ROC curves, loss tracking
- **Clean Code**: Well-documented functions for training, evaluation, and visualization

---

## Why This Matters for Portfolios

This project demonstrates:
1. **End-to-end ML pipeline**: Data → features → models → evaluation
2. **Advanced probabilistic modeling**: Not just sklearn, but custom Bayesian design
3. **Thoughtful feature engineering**: Domain knowledge applied to NLP + sentiment analysis
4. **Rigorous evaluation**: Multiple metrics, validation discipline, early stopping
5. **Communication**: Clear documentation, visualizations, and narrative explanation

---

## References & Libraries

- **Pyro**: Probabilistic programming library (pyro-ppl.org)
- **PyTorch**: Deep learning framework
- **Scikit-learn**: ML preprocessing and evaluation
- **VADER Sentiment**: Lexicon-based sentiment analysis
- **LDA (sklearn)**: Latent Dirichlet Allocation implementation

---

## License

This project is provided as-is for educational and portfolio purposes.

---

## Questions?

If you run the notebook and have questions, check:
1. Do you have the data files in `data/`?
2. Are all dependencies installed? Run `pip list | grep -E "torch|pyro|sklearn|pandas"`
3. Does your GPU have enough memory? (8GB+ recommended for full dataset)

Feel free to modify the notebook for experiments or adjust hyperparameters for faster runs!