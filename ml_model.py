from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA


ds = load_dataset("jasonhwan/yc-startup-pitches-with-scores")

df = pd.DataFrame(ds)

data = pd.json_normalize(df['train'])
data.dropna(subset=['transcript'], inplace=True)

X = data['transcript']
y = data[['clarity_score', 'team_market_fit_score', 'traction_validation_score']]

# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert transcript into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



pca = PCA(n_components=100)  # Reduce to 100 features
X_train_tfidf_reduced = pca.fit_transform(X_train_tfidf.toarray())
X_test_tfidf_reduced = pca.transform(X_test_tfidf.toarray())


# Define the model
model = MultiOutputClassifier(LinearSVC())

# Define cross-validation 
cv = KFold(n_splits=5, shuffle=True, random_state=42)

qwk_train_scores = {"clarity": [], "team_market_fit": [], "traction_validation": []}
qwk_val_scores = {"clarity": [], "team_market_fit": [], "traction_validation": []}

# Perform cross-validation manually
for train_idx, test_idx in cv.split(X_train_tfidf_reduced):
    X_fold_train, X_fold_test = X_train_tfidf_reduced[train_idx], X_train_tfidf_reduced[test_idx]
    y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

    # Train model
    model.fit(X_fold_train, y_fold_train)

    # Predict on training and validation data
    y_fold_train_pred = model.predict(X_fold_train)  
    y_fold_test_pred = model.predict(X_fold_test)    

    # Compute QWK scores for training and validation
    for key in ["clarity", "team_market_fit", "traction_validation"]:
        qwk_train_scores[key].append(cohen_kappa_score(y_fold_train[key + "_score"], y_fold_train_pred[:, list(qwk_train_scores.keys()).index(key)], weights="quadratic"))
        qwk_val_scores[key].append(cohen_kappa_score(y_fold_test[key + "_score"], y_fold_test_pred[:, list(qwk_val_scores.keys()).index(key)], weights="quadratic"))

# Compute mean QWK scores across folds
qwk_train_macro = np.mean([np.mean(qwk_train_scores[key]) for key in qwk_train_scores])
qwk_val_macro = np.mean([np.mean(qwk_val_scores[key]) for key in qwk_val_scores])

print(f"Training Macro Average QWK Score: {qwk_train_macro:.4f}")
print(f"Validation Macro Average QWK Score: {qwk_val_macro:.4f}")

print("Columns in y_test:", y_test.columns)


# Predict on the test set
y_test_pred = model.predict(X_test_tfidf_reduced)

# Compute QWK scores for the test set
qwk_test_scores = {"clarity": [], "team_market_fit": [], "traction_validation": []}

for key in ["clarity", "team_market_fit", "traction_validation"]:
    qwk_test_scores[key].append(cohen_kappa_score(y_test[key + "_score"], y_test_pred[:, list(qwk_test_scores.keys()).index(key)], weights="quadratic"))

# Compute mean QWK score for the test set
qwk_test_macro = np.mean([np.mean(qwk_test_scores[key]) for key in qwk_test_scores])

print(f"Test Macro Average QWK Score: {qwk_test_macro:.4f}")
