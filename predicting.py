import joblib


model=joblib.load("spam_model.pkl")

vectorizer=joblib.load("vectorizer.pkl")
new = "Meeting Confirmation for Project UpdateFrom: jessica.smith@companyname.comBody: Hi [Your Name], I hope this message finds you well!..."

# Transform the new text using the loaded vectorizer
new_vector = vectorizer.transform([new]).toarray()

# Predict
prec = model.predict(new_vector)

# Print the result
if prec == 0:
    print("Not Spam")
else:
    print("Spam")
