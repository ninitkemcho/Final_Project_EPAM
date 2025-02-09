import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from scipy.sparse import save_npz

import nltk
nltk.download('punkt')
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('omw-1.4')

# Importing data
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','data', 'raw'))
file_path_train = os.path.join(base_dir, 'train.csv')
file_path_test = os.path.join(base_dir, 'test.csv')

train = pd.read_csv(file_path_train)
test = pd.read_csv(file_path_test)

# Cleaning data
train.drop_duplicates(inplace = True)

lemmatizer = WordNetLemmatizer()

def clean_review(text):
  text = text.lower()  # Lowercasing
  text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
  text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
  text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters & punctuation
  text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespaces

  text = word_tokenize(text) # Tokenization

  stop_words = set(stopwords.words("english")) # Removing stop words
  text = [word for word in text if word not in stop_words]
  text = ' '.join(text)
  #text = " ".join([stemmer.stem(word) for word in text.split()])  # Stemming
  text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemma

  return text

# Splitting as X and y
X_train = train['review'].apply(clean_review)
X_test = test['review'].apply(clean_review)

y_train = (train['sentiment'] == 'positive').astype(int)
y_test = (test['sentiment'] == 'positive').astype(int)

# Saving transformed data locally
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','data', 'processed'))

os.makedirs(base_dir, exist_ok=True)

fp_X_train = os.path.join(base_dir, 'X_train.csv')
fp_X_test = os.path.join(base_dir, 'X_test.csv')
fp_y_train = os.path.join(base_dir, 'y_train.csv')
fp_y_test = os.path.join(base_dir, 'y_test.csv')

X_train.to_csv(fp_X_train, index=False)
X_test.to_csv(fp_X_test, index=False)
y_train.to_csv(fp_y_train, index=False)
y_test.to_csv(fp_y_test, index=False)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Saving vectorized datasets for later
X_train_path = os.path.join(base_dir, "X_train_tfidf.npz")
X_test_path = os.path.join(base_dir, "X_test_tfidf.npz")

save_npz(X_train_path, X_train_tfidf)
print(f"TF-IDF train data saved to {X_train_path}")

save_npz(X_test_path, X_test_tfidf)
print(f"TF-IDF test data saved to {X_test_path}")

# Fitting Logistic Regression
log_reg = LogisticRegression(random_state = 42)
log_reg.fit(X_train_tfidf, y_train)

#Saving model locally
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", 'models'))
model_path = os.path.join(output_dir, "model.pkl")

os.makedirs(output_dir, exist_ok=True)

with open(model_path, "wb") as f:
    pickle.dump(log_reg, f)

print(f"Model saved to {model_path}")

