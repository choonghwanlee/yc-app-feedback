# Automated Evaluation of Startup Pitches

# 0. Project Overview

While many founders start their entrepreneurial journies with high hopes, 90% of startups will unfortunately fail. Having a well-crafted pitch increases a founder's chance of succeeding by helping them get more investments. Unfortunately, it is very difficult to get personalized feedback on your startup pitch. You either need a mentor from the industry, or use existing coaching services that are quite expensive.

To mitigate this issue, we are developing an app that evaluates a founder's startup pitch automatically using AI. We specifically focus on YCombinator application videos for now since YC is the world's leading and most prominent accelerator for early-stage startups.

## Key Features

- **Novel Dataset:** We personally scraped, transcribed, and scored close to 500 YC application videos across numerous batches. Transcription was done with a Whisper model, while scoring was done with GPT-4o acting as an LLM-as-a-judge. We evaluated pitches across 3 different categories: clarity, team-market fit, and traction/validation. These criterias were chosen based off of publicly available advice from VCs and YCombinator partners.
- **Model Approaches:**
  - **Naive Mean Model:** The naive mean is a random selection model that serves as the baseline (low bar) of performance for this task.
  - **Non-Deep Learning Models:** A classical machine learning model that doesn't rely on deep learning techniques.
  - **Deep Learning Models:** A multi-task BERT-based model that jointly predicts clarity, team-market fit, and traction using fine-tuned contextual embeddings.
- **Real-World Application:** A user-friendly web application where users can upload their praticing videos and get their score on the pitch.

## Evaluation Metric: Quadratic Weighted Kappa (QWK)

Instead of using for a traditional classification evaluation metric like F1, we opted for Quadratic Weighted Kappa (QWK). The reasons were quite intuitive:

1. Scores are ordinal: there's a natural order to our class labels. A score of 5 is the best, while a score of 1 is the worst. This means that a score of 5 predicted as a 2 should be penalized **more** than a prediction of 4!
2. F1 scores assume that class labels are independent of one another, which is clearly not the case
3. QWK is designed to measure inter-rater agreement between two sets of ordinal ratings, which is precisely what we have at hand!

QWK has a range of -1 to 1, where 1 is perfect agreement and scores below 0 signify random chance.

# 1. Running Instruction

- Create venv `python -m venv .venv`
- Activate venv `source .venv/bin/activate`
- Install packages `pip install -r requirements.txt`

To run the Streamlit demo locally, run `streamlit run app.py`

(fill out more after reorganize the files)

# 3. Approaches

## Naive Mean Model

To create a baseline for the task, we use a simple & naive approach to assigning scores. For each sample we evaluate, we randomly assign scores based on the probability distribution of scores in the training set.

For example, if the training set had a [0.1, 0.2, 0.3, 0.3, 0.1] probability distribution for clarity, then there'd be a 30% chance we assign a transcript in the evaluation script with a score of 3 or 4.

This naive approach yields a QWK score of 0. This intuitively makes sense since a score of 0 signifies random chance.

Thus, we want to make sure that our ML and DL models have a QWK greater than our baseline score of 0.0.

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

| Model Name | Training Set | Validation Set | Testing Set |
| ---------- | ------------ | -------------- | ----------- |
| LinearSVC  | 0.7210       | 0.3333         | 0.3533      |

## Deep Learning Models

### **Bert Model Architecture:**

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
| BERT (DL) | 0.60 | 0.49 |

### **Llama3 with Chain of Thought Distillation:**

Instead of simply fine-tuning a classifier to directly predict the score, we experiment with distilling the reasoning trace of our GPT-4o judge into a smaller language model like the 3B parameter version of Llama3.2. The motivation is that by fine-tuning Llama3.2 to generate both rationale AND score, we'd condition the predicted score on a step-by-step reasoning trace at inference time. This'd lead to less hallucination and more accurate responses.

**Methodology**

1. Chain-of-thought prompting on LLM-as-a-judge to generate rationales for each score
2. Fine-tuning Llama3.2 with Unsloth using 4-bit quantization and Low Rank Adaptation
3. Trained for 3 epochs, model with lowest model selected

**Final Results (QWK):**  
| Model Name | Validation Set |
|------------|---------------|
| Naive | 0.0 |
| LinearSVC | 0.35 |
| BERT (DL) | 0.49 |
| Llama3.2 + CoT (DL) | 0.72 |

We note that distilling reasoning into a small language model yielded superior results to other approaches, trumping the QWR of the next best model by 48%.

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
