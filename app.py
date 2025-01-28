from flask import Flask, request, render_template
import joblib
import nltk
from nltk.corpus import stopwords
import string

# Download NLTK stopwords
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('sms_spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    processed_message = preprocess_text(message)
    message_tfidf = vectorizer.transform([processed_message])
    prediction = model.predict(message_tfidf)
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template('index.html', prediction_text=f'Prediction: {result}', message=message)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)