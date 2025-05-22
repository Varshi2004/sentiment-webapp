from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
with open('best_imdb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('best_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['review']
        vect_text = vectorizer.transform([text])
        prediction = model.predict(vect_text)[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        return render_template('index.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
