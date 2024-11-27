from flask import Flask, request, render_template
import joblib

# Load the model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['email_text']
        
        # Transform input text
        email_vector = vectorizer.transform([email_text]).toarray()
        
        # Predict
        prediction = model.predict(email_vector)
        
        # Render result
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template('index.html', prediction_text=f"Result: {result}")

    
if __name__=="__main__":
    app.run(debug=True)