import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import nltk
from nltk.corpus import stopwords
import string

# Step 1: Download NLTK stopwords
nltk.download('stopwords')

# Step 2: Load the dataset
df = pd.read_csv('data/spammsg.csv')

# Step 3: Rename relevant columns
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# Drop unnecessary columns
df = df[['label', 'message']]

# Step 4: Convert labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 5: Text preprocessing
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to the 'message' column
df['message'] = df['message'].apply(preprocess_text)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 7: Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 most frequent words
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit and transform the training data
X_test_tfidf = vectorizer.transform(X_test)  # Transform the testing data

# Step 8: Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 9: Save the model and vectorizer
joblib.dump(model, 'sms_spam_model.pkl')  # Save the trained model
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  # Save the vectorizer

print("Model and vectorizer saved!")
