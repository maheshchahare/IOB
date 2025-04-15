# np_chunker_model.py
import nltk
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.classify import NaiveBayesClassifier


def npchunk_features(sentence, i, history):
    word, pos, _ = sentence[i]
    prev_tag = history[-1] if history else "START"
    next_word, next_pos = (
        (sentence[i + 1][0], sentence[i + 1][1])
        if i + 1 < len(sentence)
        else ("END", "END")
    )
    return {
        "word": word,
        "pos": pos,
        "prev_tag": prev_tag,
        "next_word": next_word,
        "next_pos": next_pos,
    }


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
        tagged_sentence = pos_tag(sentence)
        history = []
        tagged_result = []
        for i, (word, pos) in enumerate(tagged_sentence):
            featureset = npchunk_features(
                [(w, p, "O") for w, p in tagged_sentence], i, history
            )
            tag = self.classifier.classify(featureset)
            tagged_result.append(((word, pos), tag))
            history.append(tag)
        return tagged_result


class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        self.tagger = ConsecutiveNPChunkTagger(train_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return conlltags2tree(conlltags)
