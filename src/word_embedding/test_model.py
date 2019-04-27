from gensim.models.fasttext import FastText
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    model = FastText.load(args.model)
    print('vocab size:', len(model.wv.vocab))
    word = input('check similarity for word: ')
    print(model.most_similar(word))
