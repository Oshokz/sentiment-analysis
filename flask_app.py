# Import required packages
from keras.models import load_model 
from keras.preprocessing.sequence import pad_sequences 
import pickle  # to load model and tokenizer
import numpy as np
from flask import Flask, request, render_template
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Create a Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("model.h5")

# Load the saved tokenizer used during training
with open("tokenizer.pkl", "rb") as tk:
    tokenizer = pickle.load(tk)

# Define the function to preprocess the user text input
def preprocess_text(text):
    # Tokenize the text
    tokens = tokenizer.texts_to_sequences([text])
    # Pad the sequences to a fixed length
    padded_tokens = pad_sequences(tokens, maxlen=100)
    return padded_tokens[0]

@app.route("/", methods=["GET", "POST"])
def predict():
    sentiment = ""
    custom_message = ""

    if request.method == "POST":
        # Get user input from the form
        user_input = request.form["user_input"]

        # Preprocess the user input
        processed_input = preprocess_text(user_input)

        # Make prediction using the loaded model
        prediction = model.predict(np.array([processed_input]))

        # Get the average prediction score
        average_prediction = prediction.mean()

        # Classify sentiment based on the average score
        if average_prediction > 0.5:
            sentiment = "Negative"
            custom_message = "It seems your sentiment is negative. Remember, tough times don't last, tough people do. Take a moment for yourself."
        else:
            sentiment = "Positive"
            custom_message = "Great news! Your sentiment is positive. Keep up the positivity!"

    # Render the template and pass the sentiment and message
    return render_template("index.html", sentiment=sentiment, custom_message=custom_message)

if __name__ == "__main__":
    app.run(debug=True)
