# Automated Evaluation of Startup Pitches
# 0. Project Overview
## Key Features

- **LLM Judge:** (fill out brief description on how we get the labels)
- **Model Approaches:**
  - **Naive Mean Model:** (Jason)
  - **Non-Deep Learning Models:** A classical machine learning model that doesn't rely on deep learning techniques. 
  - **Deep Learning Models:** A multi-task BERT-based model that jointly predicts clarity, team-market fit, and traction using fine-tuned contextual embeddings.
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
- **Results(QWK)**

| Model Name  | Training Set |Validation Set |Testing Set |
| ----------- | -------- |-------- |-------- |
| LinearSVC    | 0.7210   |0.3333   |0.3533   |


## Deep Learning Models

** Bert Model Architecture:**  
- Uses `bert-base-uncased` as the encoder backbone with gradient checkpointing to reduce memory usage.  
- Applies a shared BERT encoder followed by three separate classifiers for Clarity, Team-Market Fit, and Traction.

**Data Handling:**  
- Inputs are tokenized and converted into PyTorch Datasets.  
- Scores are normalized to 0-based class labels (for 5-class classification).  

**Training & Optimization:**  
- Multi-task loss using weighted sum of CrossEntropyLoss for the three outputs.  
- Optimized using AdamW and StepLR scheduler.  
- Best model checkpoint selected based on highest average QWK score on the validation set.

**Evaluation Metric:**  
- Quadratic Weighted Kappa (QWK) computed individually for each output and averaged.  
- Used to compare performance across training, validation, and test sets.

**Results (QWK):**  
| Model Name | Training Set | Validation Set | 
|------------|---------------|----------------|
| BERT (DL)  | 0.60          | 0.49            |

# Application

## Demo Link
We have two different links for the application, link 1 is based on the better version of model, and link 2 is based on another version of model. Since using the better model would be more costly so it would be taken down after the demo day.
[**(Link 1)**]()
[**(Link 2)**](https://huggingface.co/spaces/yiqing111/yc_app_feedback)

## Run Streamlit app locally

To run the code, run the following command:

```bash
streamlit run app.py
```
Click on the Local URL (http://localhost:8501) to open the Streamlit app in your browser.


  
