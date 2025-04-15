import gzip
import pickle
import nltk
from nltk import pos_tag
from nltk.chunk import tree2conlltags
from nltk.tree import Tree
from np_chunker_model import ConsecutiveNPChunker  # ✅ Import from shared module

# Load ATIS dataset
filename = "atis.fold0.pkl.gz"
with gzip.open(filename, "rb") as f:
    try:
        train_set, valid_set, test_set, dicts = pickle.load(f, encoding="latin1")
    except Exception:
        f.seek(0)
        train_set, valid_set, test_set, dicts = pickle.load(f)

train_x, _, train_label = train_set
test_x, _, test_label = test_set

words = dicts["words2idx"]
labels = dicts["labels2idx"]

id_to_words = {words[k]: k for k in words}
id_to_labels = {labels[k]: k for k in labels}

# Decode sentence ids into words
train_sents_tokens = [[id_to_words[id] for id in sent] for sent in train_x]


# Convert to Tree structure for chunking
def convert_to_tree(tokens):
    tagged_words = pos_tag(tokens)
    np = [
        (word, tag)
        for word, tag in tagged_words
        if tag in ("DT", "JJ", "NN", "NNS", "NNP", "NNPS")
    ]
    remaining = [(word, tag) for word, tag in tagged_words if (word, tag) not in np]
    tree = (
        Tree("S", [Tree("NP", np)] + [(word, tag) for word, tag in remaining])
        if np
        else Tree("S", tagged_words)
    )
    return tree


train_trees = [convert_to_tree(sent) for sent in train_sents_tokens]

# Convert trees to IOB format
train_conll = []
for tree in train_trees:
    try:
        train_conll.append(tree2conlltags(tree))
    except IndexError as e:
        print("Error in tree2conlltags:", tree)
        raise e

# Train the model using the shared class
chunker = ConsecutiveNPChunker(train_conll)

# Save the model
with open("NBmodel_new.pkl", "wb") as f:
    pickle.dump(chunker, f)

print("✅ Model trained and saved successfully as 'NBmodel_new.pkl'")
