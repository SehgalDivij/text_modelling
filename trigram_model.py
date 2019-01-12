from nltk.corpus import brown
from nltk import trigrams, ngrams
from collections import Counter, defaultdict
import random

def model_trigram(sentences):
    """
        Create a trigram model for sentences present in the model.
    """
    model = defaultdict(lambda: defaultdict(lambda: 0))

    # Estimate counts for each word in association with a Trigram.
    for sentence in sentences:
        for w1, w2, w3 in ngrams(sentence, 3, pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1

    return model


def model_proabilities(model):
    """
            Assign probabilities to chance of occurence of each string.
            Works with a trigram model.
    """
    # Assign probabilties for each word to be followed by two.
    for w1_w2_w3 in model:
        total_count = float(sum(model[w1_w2_w3].values()))
        for w4 in model[w1_w2_w3]:
            model[w1_w2_w3][w4] /= total_count

    return model


def generate_sentence(model):
    """
            Generate random sentence from the learned model.
    """
    text = [None, None, None]
    sentence_finished = False

    # generate random  sentences
    while not sentence_finished:
        r = random.random()
        accumulator = .0

        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]
            if accumulator >= r:
                text.append(word)
                break

        if text[-2:] == [None, None]:
            sentence_finished = True

    return ' '.join([t for t in text if t])


if __name__ == '__main__':
    print('Modelling the corpus')
    model = model_trigram(brown.sents())

    print('Assign probabilities.')
    model = model_proabilities(model)

    print('Generating sentences from the model')
    print(generate_sentence(model))
