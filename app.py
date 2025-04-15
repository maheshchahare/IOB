import nltk
import pickle
from nltk.chunk import tree2conlltags
from flask import Flask, request, render_template

# ✅ Import model class definitions from shared module
from np_chunker_model import ConsecutiveNPChunker, ConsecutiveNPChunkTagger

nltk.data.path.append("./nltk_data")

# Initialize Flask app
app = Flask(__name__)

# Load the trained Naive Bayes chunker model
with open("NBmodel.pkl", "rb") as f:
    loaded_chunker = pickle.load(f)
print("✅ Model loaded successfully!")


# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        sentence = request.form["sentence"]
        if sentence:
            # Tokenize and parse the input sentence
            words = nltk.word_tokenize(sentence)
            chunked_tree = loaded_chunker.parse(words)
            iob_tags = tree2conlltags(chunked_tree)

            # Prepare result string for rendering
            result = "\n".join(
                [f"{word} ({pos}) → {chunk}" for word, pos, chunk in iob_tags]
            )
            return render_template("index.html", result=result)
    return render_template("index.html", result="")


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

# ✅ Run using:
# waitress-serve --listen=localhost:8000 app:app
