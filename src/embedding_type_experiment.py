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
    parser.add_argument('--general_embedding_model', default='../data/word_embedding/general_embedding_300.model')
    parser.add_argument('--domain_embedding_model', default='../data/word_embedding/domain_embedding_100.model')
    parser.add_argument('--merged_embedding_model', default='../data/word_embedding/merged_embedding_300.model')
    parser.add_argument('--general_embedding_dim', type=int, default=300)
    parser.add_argument('--domain_embedding_dim', type=int, default=100)
    parser.add_argument('--merged_embedding_dim', type=int, default=300)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--patience', type=int, default=1)

    args = parser.parse_args()

    batch = False if args.batch_size == 1 else True

    embedding_types = [
        'double_embedding',
        'general_embedding',
        'domain_embedding',
        'merged_embedding'
    ]

    X, y = load_data(args.train_data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    feature_extractor = FeatureExtractor(args.general_embedding_model, args.domain_embedding_model, args.merged_embedding_model
                                         args.general_embedding_dim, args.domain_embedding_dim, args.merged_embedding_dim)

    extractor = AspectOpinionExtractor()

    for embedding_type in embedding_types:
        X_train2, y_train2 = prep_train_data(X_train, y_train, feature_extractor, embedding_type, batch)
        X_val3 = feature_extractor.get_features(X_val, embedding_type)
        X_val2, y_val2 = prep_train_data(X_val, y_val, feature_extractor, embedding_type, batch)

        if embedding_type == 'double_embedding':
            input_size = args.general_embedding_dim + args.domain_embedding_dim
        elif embedding_type == 'general_embedding':
            input_size = args.general_embedding_dim
        elif embedding_type == 'domain_embedding':
            input_size = args.args.domain_embedding_dim
        elif embedding_type == 'merged_embedding':
            input_size = args.merged_embedding_dim

        print('Embedding type:', embedding_type)
        extractor.init_model(input_size=input_size, rnn_type='bilstm')
        print(extractor.get_summary())
        start_time = time.time()
        np.random.seed(42)
        extractor.fit(X_train2, y_train2, X_val2, y_val2, (embedding_type + '.mdl'),
                      epoch=args.epoch, batch_size=args.batch_size,
                      verbose=args.verbose, patience=args.patience)
        finish_time = time.time()
        print('Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))
        extractor = AspectOpinionExtractor.load(embedding_type + '.mdl')
        extractor.evaluate(X_val3, y_val)
        print()
