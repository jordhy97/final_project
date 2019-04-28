from aoe.aspect_opinion_extractor import AspectOpinionExtractor
from aoe.feature_extractor import FeatureExtractor
from datetime import timedelta
from sklearn.model_selection import train_test_split
from utils import load_data, prep_train_data
import argparse
import numpy as np
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='../data/reviews/train.txt')
    parser.add_argument('--general_embedding_model', default='../data/word_embedding/general_embedding.vec')
    parser.add_argument('--domain_embedding_model', default='../data/word_embedding/domain_embedding_100.model')
    parser.add_argument('--general_embedding_dim', type=int, default=300)
    parser.add_argument('--domain_embedding_dim', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--patience', type=int, default=1)

    args = parser.parse_args()

    batch = False if args.batch_size == 1 else True

    rnn_types = [
        'gru',
        'lstm',
        'bilstm',
        'bigru'
    ]

    X, y = load_data(args.train_data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    feature_extractor = FeatureExtractor(args.general_embedding_model, args.domain_embedding_model,
                                         args.general_embedding_dim, args.domain_embedding_dim)

    X_train, y_train = prep_train_data(X_train, y_train, feature_extractor, batch)
    X_val, y_val2 = prep_train_data(X_val, y_val, feature_extractor, batch)

    input_size = args.general_embedding_dim + args.domain_embedding_dim
    extractor = AspectOpinionExtractor()

    for rnn_type in rnn_types:
        print('RNN type:', rnn_type)
        extractor.init_model(input_size=input_size, rnn_type=rnn_type)
        print(extractor.get_summary())
        start_time = time.time()
        np.random.seed(42)
        extractor.fit(X_train, y_train, X_val, y_val2, (rnn_type + '.mdl'),
                      epoch=args.epoch, batch_size=args.batch_size,
                      verbose=args.verbose, patience=args.patience)
        finish_time = time.time()
        print('Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))
        extractor = AspectOpinionExtractor.load(rnn_type + '.mdl')
        extractor.evaluate(X_val, y_val)
        print()