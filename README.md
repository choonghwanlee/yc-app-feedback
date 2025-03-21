# yc-app-feedback
# 0. Project Overview
## Key Features

- **LLM Judge:** (fill out brief description on how we get the labels)
- **Model Approaches:**
  - **Naive Mean Model:** (Jason)
  - **Non-Deep Learning Models:** A classical machine learning model that doesn't rely on deep learning techniques. 
  - **Deep Learning Models:** (Yiqing)
- **Real-World Application:** A user-friendly web application where users can upload their praticing videos and get their score on the pitch.

## Evaluation Metric: Quadratic Weighted Kappa (QWK)
(fill out reasons)
# 1. Running Instruction

- Create venv `python -m venv .venv`
- Activate venv `source .venv/bin/activate`
- Install packages `pip install -r requirements.txt`

To run the Streamlit demo locally, run `streamlit run app.py`

(fill out more after reorganize the files)

# 3. Approaches

## Naive Mean Model

(Jason)

## Non-Deep Learning Models


## Deep Learning Models

- **Feature Extraction:**
  - Converts the textual data into numerical features using TfidfVectorizer.
  - Applies PCA to reduce dimensionality of the TF-IDF features.
- **Model Definition:**
  - Uses MultiOutputClassifier with a LinearSVC as the base model for multi-output classification.
- **Cross-Validation:**
  - Performs 5-fold cross-validation using KFold, training the model on each fold and computing the Quadratic Weighted Kappa (QWK) for both training and validation sets.
- **Evaluation:**
  - Computes and prints the macro-average QWK score for both training and validation sets.
  - Predicts on the test set and computes the QWK score for the test set as well.
  
