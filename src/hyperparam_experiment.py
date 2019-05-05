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
    parser.add_argument('--general_embedding_dim', type=int, default=300)
    parser.add_argument('--domain_embedding_dim', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--patience', type=int, default=1)

    args = parser.parse_args()

    batch = False if args.batch_size == 1 else True

    n_hiddens = [25, 50, 75]
    n_tensors = [10, 15, 20]
    n_layers = [1, 2, 3]
    dropout_rates = [0, 0.2, 0.5]

    X, y = load_data(args.train_data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    feature_extractor = FeatureExtractor(args.general_embedding_model, args.domain_embedding_model,
                                         general_dim=args.general_embedding_dim, domain_dim=args.domain_embedding_dim)

    X_train, y_train = prep_train_data(X_train, y_train, feature_extractor, batch=batch)
    X_val2 = feature_extractor.get_features(X_val)
    X_val, y_val2 = prep_train_data(X_val, y_val, feature_extractor, batch=batch)

    input_size = args.general_embedding_dim + args.domain_embedding_dim
    extractor = AspectOpinionExtractor()

    count = 0

    for n_layer in n_layers:
        for n_tensor in n_tensors:
            for n_hidden in n_hiddens:
                for dropout_rate in dropout_rates:
                    count += 1
                    print('n hidden:', n_hidden)
                    print('n tensor:', n_tensor)
                    print('n layer:', n_layer)
                    print('dropout rate:', dropout_rate)

                    extractor.init_model(input_size=input_size,
                                         n_hidden = n_hidden,
                                         n_tensors = n_tensor,
                                         n_layers = n_layer,
                                         dropout_rate = dropout_rate,
                                         rnn_type = 'bilstm')

                    print(extractor.get_summary())
                    start_time = time.time()
                    np.random.seed(42)
                    extractor.fit(X_train, y_train, X_val, y_val2, ('param' + str(count) + '.mdl'),
                                  epoch=args.epoch, batch_size=args.batch_size,
                                  verbose=args.verbose, patience=args.patience)
                    finish_time = time.time()
                    print('Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))
                    extractor = AspectOpinionExtractor.load('param' + str(count) + '.mdl')
                    extractor.evaluate(X_val2, y_val)
                    print()
