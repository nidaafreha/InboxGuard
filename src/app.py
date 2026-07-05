from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__, template_folder='../templates')

model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)[0]
    result = "🚨 SPAM!" if prediction == 1 else "✅ NOT SPAM!"
    return render_template('index.html', result=result, email=email)

if __name__ == '__main__':
    app.run(debug=True)