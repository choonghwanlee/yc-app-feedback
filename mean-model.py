from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import numpy as np


def naive_baseline_predict(train_df, test_df, label_cols, N=1000):
    """
    Predicts labels for the test set by randomly sampling from the class distribution in the training set N times
    
    Parameters:
        train_df (pd.DataFrame): Training dataset containing the labels.
        test_df (pd.DataFrame): Test dataset (same columns as train_df, but labels will be predicted).
        label_cols (list): List of column names corresponding to the labels.
        N (int): Number of times to randomly sample each prediction.
    
    Returns:
        pd.DataFrame: Test dataset with predicted labels.
    """
    predictions = test_df.copy()
    
    for label in label_cols:
        # Compute class distribution in training set
        class_probs = train_df[label].value_counts(normalize=True).sort_index()
        
        # Sample N times
        sampled_predictions = np.array([
            np.random.choice(class_probs.index, size=len(test_df), p=class_probs.values)
            for _ in range(N)
        ])
        
        # Take the mode (most frequent value) across the N samples for each test instance
        predictions[label] = [np.bincount(sampled_predictions[:, i]).argmax() for i in range(len(test_df))]
    
    return predictions


if __name__ == '__main__':
    dataset = load_dataset("jasonhwan/yc-startup-pitches-with-scores")

    full_dataset = dataset['train']

    df = pd.DataFrame(full_dataset)

    X = df['transcript']
    y = df[['clarity_score', 'team_market_fit_score', 'traction_validation_score']]

    _, _, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictions = naive_baseline_predict(y_train, y_test, ['clarity_score', 'team_market_fit_score', 'traction_validation_score'])

    kappa_scores = []
    for label in ['clarity_score', 'team_market_fit_score', 'traction_validation_score']:
        kappa = cohen_kappa_score(y_test[label], predictions[label], weights='quadratic')
        print(f"QWK score for {label}: {kappa}")
        kappa_scores.append(kappa)

    print(f"Average QWK Score: {np.mean(kappa_scores)}")