import numpy as np
from gensim.models.fasttext import FastText

class FeatureExtractor():
    """
    Feature extractor for aspect and Opinion Terms extraction.

    Uses double embeddings adapted from the paper
    "Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction" published in ACL 2018.
    """
    def __init__(self, general_embedding_model=None, domain_embedding_model=None, merged_embedding_model=None,
                 general_dim=0, domain_dim=0, merged_dim=0):
        # self.general_embedding = dict()
        # with open(general_embedding_model) as f:
           # for l in f:
               # rec = l.rstrip().split(' ')
               # self.general_embedding[rec[0]] = np.asarray([float(r) for r in rec[1:]])
        if general_embedding_model is not None:
            self.general_embedding = FastText.load(general_embedding_model)
            self.general_unknown = np.zeros(general_dim)

        if domain_embedding_model is not None:
            self.domain_embedding = FastText.load(domain_embedding_model)
            self.domain_unknown = np.zeros(domain_dim)

        if merged_embedding_model is not None:
            self.merged_embedding = FastText.load(merged_embedding_model)
            self.merged_unknown = np.zeros(merged_dim)

    def get_max_len(self, X):
        """
        Get the maximum sequence length from the given data.

        Parameters
        ----------
        X: List of sequence data.

        Returns
        -------
        Maximum sequence length.
        """
        max_len = 0
        for seq in X:
            max_len = max(max_len, len(seq))
        return max_len

    def get_general_embedding(self, tokens, max_len=None):
        """
        Transform sequence of tokens into sequence of vectors of general embedding.

        Parameters
        ----------
        tokens: list
            Tokens to transform.

        max_len: int, default: None
            Maximum length of the sequence (used for padding), None if padding is not used.

        Returns
        -------
        sequence of vectors of the tokens.
        """
        result = []
        for token in tokens:
           # if token in self.general_embedding:
           #     general_embedding = self.general_embedding[token]
           # else:
           #     general_embedding = self.general_unknown
            try:
                general_embedding = self.general_embedding.wv[token]
            except:
                general_embedding = self.general_unknown

            result.append(general_embedding)

        if max_len != None:
            for i in range(len(tokens), max_len):
                result.append(self.general_unknown)

        return np.asarray(result)

    def get_domain_embedding(self, tokens, max_len=None):
        """
        Transform sequence of tokens into sequence of vectors of domain embedding.

        Parameters
        ----------
        tokens: list
            Tokens to transform.

        max_len: int, default: None
            Maximum length of the sequence (used for padding), None if padding is not used.

        Returns
        -------
        sequence of vectors of the tokens.
        """
        result = []
        for token in tokens:
            try:
                domain_embedding = self.domain_embedding.wv[token]
            except:
                domain_embedding = self.domain_unknown

            result.append(domain_embedding)

        if max_len != None:
            for i in range(len(tokens), max_len):
                result.append(self.domain_unknown)

        return np.asarray(result)

    def get_merged_embedding(self, tokens, max_len=None):
        """
        Transform sequence of tokens into sequence of vectors of merged embedding.

        Parameters
        ----------
        tokens: list
            Tokens to transform.

        max_len: int, default: None
            Maximum length of the sequence (used for padding), None if padding is not used.

        Returns
        -------
        sequence of vectors of the tokens.
        """
        result = []
        for token in tokens:
            try:
                merged_embedding = self.merged_embedding.wv[token]
            except:
                merged_embedding = self.merged_unknown

            result.append(merged_embedding)

        if max_len != None:
            for i in range(len(tokens), max_len):
                result.append(self.merged_unknown)

        return np.asarray(result)

    def get_double_embedding(self, tokens, max_len=None):
        """
        Transform sequence of tokens into sequence of vectors of double embedding.

        Parameters
        ----------
        tokens: list
            Tokens to transform.

        max_len: int, default: None
            Maximum length of the sequence (used for padding), None if padding is not used.

        Returns
        -------
        sequence of vectors of the tokens.
        """
        result = []
        for token in tokens:
           # if token in self.general_embedding:
           #     general_embedding = self.general_embedding[token]
           # else:
           #     general_embedding = self.general_unknown
            try:
                general_embedding = self.general_embedding.wv[token]
            except:
                general_embedding = self.general_unknown

            try:
                domain_embedding = self.domain_embedding.wv[token]
            except:
                domain_embedding = self.domain_unknown

            result.append(np.concatenate((general_embedding, domain_embedding)))

        if max_len != None:
            for i in range(len(tokens), max_len):
                result.append(np.concatenate((self.general_unknown, self.domain_unknown)))

        return np.asarray(result)


    def transform(self, tokens, feature='double_embedding', max_len=None):
        """
        Transform sequence of tokens into sequence of vectors.

        Parameters
        ----------
        tokens: list
            Tokens to transform.

        max_len: int, default: None
            Maximum length of the sequence (used for padding), None if padding is not used.

        feature: feature to get, default: 'double_embedding'
            Feature to use.
            available options: {'double_embedding', 'general_embedding', 'domain_embedding', 'merged_embedding'}

        Returns
        -------
        sequence of vectors of the tokens.
        """
        if feature == 'double_embedding':
            return self.get_double_embedding(tokens, max_len)
        elif feature == 'general_embedding':
            return self.get_general_embedding(tokens, max_len)
        elif feature == 'domain_embedding':
            return self.get_domain_embedding(tokens, max_len)
        elif feature == 'merged_embedding':
            return self.get_merged_embedding(tokens, max_len)
        else:
            raise ValueError("Unknown feature type.")

    def get_oov(self, tokens):
        """
        Get the tokens that are OOV (out of vocabulary, tokens are not defined in the word embedding)

        Parameters
        ----------
        tokens: list
            Tokens to check for OOV.

        Returns
        -------
        tuple of list of OOV tokens for general and domain embedding.
        """
        general_oov = []
        domain_oov = []
        merged_oov = []
        for token in tokens:
           # if token not in self.general_embedding:
           #     general_oov.append(token)
            try:
                general_embedding = self.general_embedding.wv[token]
            except:
                general_oov.append(token)
            try:
                domain_embedding = self.domain_embedding.wv[token]
            except:
                domain_oov.append(token)
            try:
                merged_embedding = self.merged_embedding.wv[token]
            except:
                merged_oov.append(token)


        return general_oov, domain_oov, merged_oov

    def get_features(self, X, feature='double_embedding', max_len=None):
        """
        Get data features.

        Parameters
        ----------
        X: list
            Data to be extracted.

        feature: feature to get, default: 'double_embedding'
            Feature to use.
            available options: {'double_embedding', 'general_embedding', 'domain_embedding', 'merged_embedding'}

        max_len: int, default: None
            Maximum length of the sequence (used for padding), None if padding is not used.

        Returns
        -------
        Aspect and Opinion Terms Extractor feature (word embeddings).
        """
        features = []
        for i in range(len(X)):
            if max_len is not None:
                features.append(self.transform(X[i], feature, max_len))
            else:
                features.append(np.asarray([self.transform(X[i], feature)]))

        return features
