from aoe.aspect_opinion_extractor import AspectOpinionExtractor
from aoe.feature_extractor import FeatureExtractor
from subprocess import call
from os import system
import argparse
import numpy as np
import tensorflow as tf

class Main():
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.input_filename = 'temp/in.txt'
        self.output_filename = 'temp/out.txt'
        self.model1 = 'model/de-cmla.mdl'
        self.model2 = 'model/no_attention.mdl'
        self.model3 = 'model/general_embedding.mdl'
        self.model4 = 'model/domain_embedding.mdl'

        general_embedding_model = '../data/word_embedding/general_embedding.vec'
        domain_embedding_model = '../data/word_embedding/domain_embedding_100.model'
        general_embedding_dim = 300
        domain_embedding_dim = 100

        self.feature_extractor = FeatureExtractor(general_embedding_model, domain_embedding_model,
                                             general_dim=general_embedding_dim, domain_dim=domain_embedding_dim)

    def preprocess(self, input_filename, output_filename):
        call('jython preprocess/preprocess.py ' + input_filename + ' ' + output_filename, shell=True)

    def get_review_tokens(self, filename):
        raw, formalized = [], []
        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if line:
                    raw_token, formalized_token = line.split('\t')
                    raw.append(raw_token)
                    formalized.append(formalized_token)

        return raw, formalized

    def extract_terms(self, review):
        with open(self.input_filename, 'w') as input_file:
            input_file.write(review)

        # preproses
        self.preprocess(self.input_filename, self.output_filename)

        # get tokens
        raw, formalized = self.get_review_tokens(self.output_filename)

        # get features
        de_features = self.feature_extractor.get_features([formalized], feature='double_embedding')
        gen_features = self.feature_extractor.get_features([formalized], feature='general_embedding')
        dom_features = self.feature_extractor.get_features([formalized], feature='domain_embedding')
        extractor = AspectOpinionExtractor.load(self.model1)
        labels = extractor.predict(de_features)[0]
        result1 = self.get_result(review, raw, labels)

        extractor = AspectOpinionExtractor.load(self.model2)
        labels = extractor.predict(de_features)[0]
        result2 = self.get_result(review, raw, labels)

        extractor = AspectOpinionExtractor.load(self.model3)
        labels = extractor.predict(gen_features)[0]
        result3 = self.get_result(review, raw, labels)

        extractor = AspectOpinionExtractor.load(self.model4)
        labels = extractor.predict(dom_features)[0]
        result4 = self.get_result(review, raw, labels)

        return result1, result2, result3, result4

    def get_result(self, review, raw, labels):
        aspects = []
        sentiments = []
        result = review

        idx = 0
        pos = 0
        while idx < len(labels):
            if labels[idx] == 'B-ASPECT' or labels[idx] == 'I-ASPECT':
                pos = result.find(raw[idx], pos)
                result = result[:pos] + '<span class="highlight_aspect">' + result[pos:]
                aspect = raw[idx]
                pos += (len(raw[idx]) + 31)
                idx += 1
                while idx < len(labels) and labels[idx] == 'I-ASPECT':
                    pos = result.find(raw[idx], pos)
                    pos += len(raw[idx])
                    aspect += (' ' + raw[idx])
                    idx += 1
                result = result[:pos] + '</span>' + result[pos:]
                pos += 7
                aspects.append(aspect)

            elif labels[idx] == 'B-SENTIMENT' or labels[idx] == 'I-SENTIMENT':
                pos = result.find(raw[idx], pos)
                result = result[:pos] + '<span class="highlight_opinion">' + result[pos:]
                sentiment = raw[idx]
                pos += (len(raw[idx]) + 32)
                idx += 1
                while idx < len(labels) and labels[idx] == 'I-SENTIMENT':
                    pos = result.find(raw[idx], pos)
                    pos += len(raw[idx])
                    sentiment += (' ' + raw[idx])
                    idx += 1
                result = result[:pos] + '</span>' + result[pos:]
                pos += 7
                sentiments.append(sentiment)

            else:
                idx += 1

        # print()
        # if aspects:
        #     print('Ekspresi Aspek:')
        #     for aspect in aspects:
        #         print(aspect)
        #     print()
        #
        # if sentiments:
        #     print('Ekspresi Sentimen:')
        #     for sentiment in sentiments:
        #         print(sentiment)
        #     print()

        return result
