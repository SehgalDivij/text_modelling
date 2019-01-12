from nltk.corpus import brown
from nltk import trigrams, ngrams
from collections import Counter, defaultdict
import random


def model_tetragram(sentences, n):
    """
            Create a trigram model for sentences present in the model.
    """
    model = defaultdict(lambda: defaultdict(lambda: 0))

    # Estimate counts for each wo(From the Brown Corpus)d in association with a Trigram.
    for sentence in sentences:
        for l in ngrams(sentence, n, pad_right=True, pad_left=True):
            l = list(l)
            model[tuple(l[0:len(l)-1])][l[len(l) - 1]] += 1
            # print(tuple(l[0:len(l)-1]), l[len(l) - 1])

    return model


def model_proabilities(model):
    """
            Assign probabilities to chance of occurence of each string.
            Works with a trigram model.
    """
    # Assign probabilties for each word to be followed by two.
    for phrase in model:
        total_count = float(sum(model[phrase].values()))
        for word in model[phrase]:
            model[phrase][word] /= total_count

    return model


def generate_sentence(model, n):
    """
        Generate random sentence from the learned model.
    """
    text = [None] * (n-1)
    sentence_finished = False

    # generate random  sentences
    while not sentence_finished:
        r = random.random()
        accumulator = .0

        for word in model[tuple(text[1-n:])].keys():
            accumulator += model[tuple(text[1-n:])][word]
            if accumulator >= r:
                text.append(word)
                break

        if text[1-n:] == [None] * (n-1):
            sentence_finished = True

    return ' '.join([t for t in text if t])


if __name__ == '__main__':
    print('N gram model(From the Brown Corpus)')
    print('Enter the number n, for which to create an n-gram.')
    print('e.g. For a Trigram model, enter 3.')
    
    n = int(input())

    print('Generating a ' + str(n) + '-gram Model')

    print('Modelling the corpus')
    model = model_tetragram(brown.sents(), n)

    print('Assign probabilities.')
    model = model_proabilities(model)

    print('Generating sentences from the model')
    print(generate_sentence(model, n))
