import time
import multiprocessing
from datetime import timedelta
import argparse
from gensim.models import word2vec
from gensim.models.fasttext import FastText

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('type', help='type of word embedding to train (domain or general)')
    args = parser.parse_args()

    start_time = time.time()
    sentences = word2vec.LineSentence(args.infile)

    print('Training FastText Model...')
    if args.type == 'domain':
        model = FastText(sentences, size=100, iter=30, workers=multiprocessing.cpu_count()-1)
    elif args.type == 'general':
        model = FastText(sentences, size=300, iter=5, negative=10, min_n=5, max_n=5, workers=multiprocessing.cpu_count()-1)

    model.save(args.outfile)
    finish_time = time.time()

    print('Finished. Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))
