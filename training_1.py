import gzip
import pickle

filename = "atis.fold0.pkl.gz"
f = gzip.open(filename, "rb")
try:
    train_set, valid_set, test_set, dicts = pickle.load(f, encoding="latin1")
except Exception as e:
    print(e)
    train_set, valid_set, test_set, dicts = pickle.load(f)
finally:
    f.close()

train_x, _, train_label = train_set
test_x, _, test_label = test_set

words = dicts["words2idx"]
labels = dicts["labels2idx"]

id_to_words = {words[k]: k for k in words}
id_to_labels = {labels[k]: k for k in labels}

w = []
for index in range(len(train_x)):
    w.append([id_to_words[id] for id in train_x[index]])

test = []
for index in range(len(test_x)):
    test.append([id_to_words[id] for id in train_x[index]])

import nltk
from nltk import pos_tag
from nltk.tree import Tree

# Download necessary NLTK resources
# nltk.download('averaged_perceptron_tagger')


def convert_to_tree(tokens):
    tagged_words = pos_tag(tokens)  # Get POS tagging

    # Find NP (Noun Phrase) - simple heuristic (determiner, adjectives, and nouns)
    np = []
    remaining = []

    for word, tag in tagged_words:
        if tag in ("DT", "JJ", "NN", "NNS", "NNP", "NNPS"):  # Group into NP
            np.append((word, tag))
        else:
            remaining.append((word, tag))

    np_tree = Tree("NP", np) if np else None  # Create NP subtree

    # Build the full sentence tree
    if np_tree:
        tree = Tree("S", [np_tree] + [Tree(tag, [word]) for word, tag in remaining])
    else:
        tree = Tree("S", [Tree(tag, [word]) for word, tag in tagged_words])

    return tree


# Convert each tokenized sentence into an nltk.Tree format
train_sents = [convert_to_tree(sent) for sent in w]

# Print the structured output
for tree in train_sents:
    # print(tree)
    break

# import nltk
# from nltk import pos_tag
# from nltk.tree import Tree
from nltk.chunk import tree2conlltags, conlltags2tree
from nltk.classify import NaiveBayesClassifier

# Download necessary resources
# nltk.download('averaged_perceptron_tagger')


# Function to convert tokenized sentences into a proper Tree format
def convert_to_tree(tokens):
    tagged_words = pos_tag(tokens)  # POS tagging

    # Group words into NP (Noun Phrase) based on simple heuristic
    np = []
    remaining = []

    for word, tag in tagged_words:
        if tag in ("DT", "JJ", "NN", "NNS", "NNP", "NNPS"):  # Group into NP
            np.append((word, tag))
        else:
            remaining.append((word, tag))

    np_tree = Tree("NP", np) if np else None  # Create NP subtree

    # Ensure the structure has only (word, POS) tuples
    if np_tree:
        tree = Tree("S", [np_tree] + [(word, tag) for word, tag in remaining])
    else:
        tree = Tree("S", [(word, tag) for word, tag in tagged_words])

    return tree


# Example tokenized sentences
# samples = [
#     ['what', 'flights', 'leave', 'atlanta', 'at', 'about', 'DIGIT', 'in', 'the', 'afternoon', 'and', 'arrive', 'in', 'san', 'francisco'],
#     ['does', 'delta', 'airlines', 'fly', 'to', 'boston']
# ]

# Convert tokenized sentences to Tree format
train_sents = [convert_to_tree(sent) for sent in w]

# 🔹 **Fix: Ensure the tree is in the correct format for `tree2conlltags()`**
train_conll = []
for tree in train_sents:
    try:
        train_conll.append(tree2conlltags(tree))
    except IndexError as e:
        print("Error with tree:", tree)
        raise e


# Feature extraction function
def npchunk_features(sentence, i, history):
    """Extract features for the classifier."""
    word, pos, chunk = sentence[i]
    prev_tag = history[-1] if history else "START"

    next_word, next_pos = (
        (sentence[i + 1][0], sentence[i + 1][1])
        if i + 1 < len(sentence)
        else ("END", "END")
    )

    features = {
        "word": word,
        "pos": pos,
        "prev_tag": prev_tag,
        "next_word": next_word,
        "next_pos": next_pos,
    }
    return features


# Define the Naive Bayes Chunk Tagger
class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            history = []
            for i, (word, pos, chunk) in enumerate(tagged_sent):
                featureset = npchunk_features(tagged_sent, i, history)
                train_set.append((featureset, chunk))
                history.append(chunk)
        self.classifier = NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        """Tag a sentence with chunk labels."""
        tagged_sentence = pos_tag(sentence)  # Ensure POS tagging is applied
        if isinstance(sentence[0], tuple):  # If input is already POS tagged
            tagged_sentence = sentence
        else:
            tagged_sentence = pos_tag(sentence)  # Perform POS tagging if not done

        history = []
        tagged_result = []
        for i, (word, pos) in enumerate(tagged_sentence):
            featureset = npchunk_features(
                [(word, pos, "O") for word, pos in tagged_sentence], i, history
            )
            tag = self.classifier.classify(featureset)
            tagged_result.append(((word, pos), tag))
            history.append(tag)
        return tagged_result


# Define the NP Chunker using Naive Bayes
class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        self.tagger = ConsecutiveNPChunkTagger(train_sents)

    def parse(self, sentence):
        """Parse input sentence into chunked structure."""
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return conlltags2tree(conlltags)


# Train the chunker
chunker = ConsecutiveNPChunker(train_conll)

# Test the trained chunker
test_sentence = "show me flights from los angeles to new york"
test_tokens = test_sentence.split()
parsed_tree = chunker.parse(test_tokens)

# Print the chunked tree
# print(parsed_tree)


iob_tags = tree2conlltags(parsed_tree)


# Function to compute accuracy in percentage
def evaluate_chunker(chunker, test_sents):
    correct = 0
    total = 0

    for gold_tree in test_sents:
        # Convert gold standard tree to IOB format
        gold_conll = tree2conlltags(gold_tree)
        words = [word for word, _, _ in gold_conll]

        # Parse the sentence (NO NEED TO POS-TAG AGAIN)
        parsed_tree = chunker.parse(words)  # <-- Pass words directly
        parsed_conll = tree2conlltags(parsed_tree)

        # Compare IOB tags
        for gold, parsed in zip(gold_conll, parsed_conll):
            _, _, gold_chunk = gold
            _, _, parsed_chunk = parsed
            total += 1
            if gold_chunk == parsed_chunk:
                correct += 1

    accuracy = (
        (correct / total) * 100 if total > 0 else 0
    )  # Calculate percentage accuracy
    return accuracy


import pickle

# Save the trained chunker
with open("NBmodel.pkl", "wb") as f:
    pickle.dump(chunker, f)

# print("Model saved successfully!")
