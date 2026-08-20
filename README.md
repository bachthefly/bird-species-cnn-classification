# Bird Species Classification with CNNs

This project explores **fine-grained bird image classification** using convolutional neural networks (CNNs) and TensorFlow.

Rather than training a single multi-class classifier, the project trains multiple binary CNN classifiers on carefully selected pairs of bird species. The resulting models are then used to characterize and classify previously unseen bird species.

## Approach

The analysis follows a two-stage classification workflow:

1. Select ten pairs of bird species for binary classification
2. Train a CNN for each species pair
3. Evaluate each binary classifier using validation data
4. Apply the trained models to two novel bird species
5. Use the resulting classifier probabilities as features
6. Train a traditional classifier to distinguish between the two novel species

## Analysis

The project includes:

- Image dataset preparation and organization
- Training/validation dataset construction with TensorFlow
- CNN-based binary image classification
- Evaluation using accuracy and confusion matrices
- Per-class precision and recall
- Prediction of novel bird species using multiple binary classifiers
- Feature construction from CNN prediction probabilities
- Final classification of novel species using scikit-learn

## Methods

**Computer Vision**
- Image preprocessing
- Image classification
- Fine-grained bird recognition

**Deep Learning**
- Convolutional Neural Networks
- TensorFlow / Keras
- Binary classification

**Machine Learning**
- Ensemble-style prediction features
- K-Nearest Neighbors / traditional classification
- Train/test evaluation

**Evaluation**
- Accuracy
- Confusion matrices
- Precision
- Recall

## Dataset

The project uses a subset of a large-scale bird image dataset containing multiple North American bird species.

Because the original dataset is too large for GitHub, the full dataset is not included in this repository.

## Repository Structure

```text
├── bird_images/              # Training/validation bird images
├── novel_species1_images/    # Images for the first novel species
├── novel_species2_images/    # Images for the second novel species
├── novel_species_analysis.ipynb                   # Model training and analysis notebooks
└── README.md

```
## Key Questions

### The project explores questions such as:

- Which bird species are most difficult to distinguish using image-based CNN classifiers?
- How well do binary classifiers generalize to previously unseen species?
- Can predictions from multiple binary classifiers provide useful representations of novel species?
- Can these representations support classification of species not included in the original binary models?

## Tools
- Python
- TensorFlow
- Keras
- scikit-learn
- pandas
- NumPy
- matplotlib
- Jupyter Notebook

## Context

This project was completed as coursework for DA 351: Advanced Descriptive Methods at Denison University.
