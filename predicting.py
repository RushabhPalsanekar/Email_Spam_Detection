import joblib


model=joblib.load("spam_model.pkl")

vectorizer=joblib.load("vectorizer.pkl")
new = """You’ve Won a $1,000 Gift Card! Claim Now! 🌟

Dear Valued Customer,

Congratulations! 🎉 You have been selected to receive a $1,000 Gift Card as part of our exclusive rewards program. This offer is available for a limited time only!

What you need to do:

Click the link below to claim your gift card.
Complete the quick verification process.
Enjoy spending your gift card at your favorite stores!
👉 Claim Your Gift Card Now!

Hurry, this offer expires in 24 hours! Don’t miss out on this incredible opportunity.

Note: This email is intended for the recipient only. If you do not claim your gift, it will be forfeited.

Warm regards,
The Rewards Team

"""

# Transform the new text using the loaded vectorizer
new_vector = vectorizer.transform([new]).toarray()

# Predict
prec = model.predict(new_vector)

# Print the result
if prec == 0:
    print("Not Spam")
else:
    print("Spam")
