from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import numpy as np


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

model = MultiOutputClassifier(RandomForestClassifier())
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# Calculate F1 score for each label separately
f1_clarity = f1_score(y_test['clarity_score'], y_pred[:, 0], average='macro', zero_division=1)
f1_team_market_fit = f1_score(y_test['team_market_fit_score'], y_pred[:, 1], average='macro', zero_division=1)
f1_traction_validation = f1_score(y_test['traction_validation_score'], y_pred[:, 2], average='macro', zero_division=1)

# Calculate macro average F1 score
f1_macro = np.mean([f1_clarity, f1_team_market_fit, f1_traction_validation])
print(f"Macro Average F1 Score: {f1_macro}")