from aoe.feature_extractor import FeatureExtractor
import numpy as np

def load_data(filename):
    """
    Loads data and label from a file.

    Parameters
    ----------
    filename: Filename that contains the data.

    File format: tab-separated, blank line at the end of a sentence.
    Example:
    ```
    pelayanan	B-ASPECT
    ramah	B-SENTIMENT
    ,	O
    kamar	B-ASPECT
    nyaman	B-SENTIMENT
    dan	O
    fasilitas	B-ASPECT
    lengkap	B-SENTIMENT
    .	O
    hanya	O
    airnya	B-ASPECT
    showernya	I-ASPECT
    kurang	B-SENTIMENT
    panas	I-SENTIMENT
    .	O
    <blank line>
    ```

    Returns
    -------
    tuple of data and labels.
    """
    data, labels = [], []
    with open(filename, encoding='utf-8') as f:
        tokens, tags = [], []
        for line in f:
            line = line.rstrip()
            if line:
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            else:
                data.append(tokens)
                labels.append(tags)
                tokens, tags = [], []

    return data, labels

def prep_train_data(X, y, feature_extractor, feature='double_embedding', batch=False):
    """
    Convert data and labels into compatible format for training.

    Parameters
    ----------
    X: Train data (list).
    y: labels (list).
    feature_extractor: FeatureExtractor.
    batch: Train in batch or not. Default: False

    Returns
    -------
    tuple of data and labels in compatible format for training.
    """
    if batch:
        max_len = feature_extractor.get_max_len(X)
    else:
        max_len = None

    X_train = feature_extractor.get_features(X, feature, max_len)

    y_train = []
    ya_train = []
    yo_train = []
    for i in range(len(y)):
        ya = []
        yo = []
        for label in y[i]:
            if label == 'O':
                ya.append([1, 0, 0])
                yo.append([1, 0, 0])
            if label == 'B-ASPECT':
                ya.append([0, 1, 0])
                yo.append([1, 0, 0])
            elif label == 'I-ASPECT':
                ya.append([0, 0, 1])
                yo.append([1, 0, 0])
            elif label == 'B-SENTIMENT':
                ya.append([1, 0, 0])
                yo.append([0, 1, 0])
            elif label == 'I-SENTIMENT':
                ya.append([1, 0, 0])
                yo.append([0, 0, 1])

        # padding
        if batch:
            for j in range(len(y[i]), max_len):
                ya.append([1, 0, 0])
                yo.append([1, 0, 0])
            ya_train.append(ya)
            yo_train.append(yo)

        else:
            ya = np.asarray([ya])
            yo = np.asarray([yo])
            y_train.append([ya, yo])

    if batch:
        return np.asarray(X_train), [np.asarray(ya_train), np.asarray(yo_train)]
    else:
        return X_train, y_train