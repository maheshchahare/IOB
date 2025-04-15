import nltk
import pickle
from nltk.chunk import tree2conlltags
from flask import Flask, request, render_template

from training_1 import ConsecutiveNPChunker, ConsecutiveNPChunkTagger

# Initialize the Flask application
app = Flask(__name__)

# Load the trained chunker model
with open("NBmodel_new.pkl", "rb") as f:
    loaded_chunker = pickle.load(f)
print("Model loaded successfully!")


# Route for the home page with the input form
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        sentence = request.form["sentence"]
        if sentence:
            # Process the sentence to get IOB tags
            words = nltk.word_tokenize(sentence)  # Tokenize input
            chunked_tree = loaded_chunker.parse(words)  # Pass only words, NOT POS-tags
            iob_tags = tree2conlltags(chunked_tree)  # Convert to IOB format
            result = "\n".join(
                [f"{word} ({pos}) → {iob}" for word, pos, iob in iob_tags]
            )
            return render_template("index.html", result=result)
    return render_template("index.html", result="")


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

# Run the file using below command:
# waitress-serve --listen=localhost:8000 app_flask:app
