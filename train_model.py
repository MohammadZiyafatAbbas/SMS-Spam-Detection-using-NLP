import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

df = pd.read_csv('data/spammsg.csv')

df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

df = df[['label', 'message']]

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['message'] = df['message'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)  
X_train_tfidf = vectorizer.fit_transform(X_train) 
X_test_tfidf = vectorizer.transform(X_test)  

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

joblib.dump(model, 'sms_spam_model.pkl')  
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  

print("Model and vectorizer saved!")
