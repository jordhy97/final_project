from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import initializers
from aoe.cmla import CMLA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report, performance_measure
import tensorflow as tf
import numpy as np

class AspectOpinionExtractor():
    """
    Aspect and Opinion Terms Extractor.

    Adapted from the paper "Coupled Multi-Layer Attentions for Co-Extraction of
    Aspect and Opinion Terms" published in AAAI 2017.
    (http://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/15-Wang-W-14441.pdf)
    The default values used in the parameters are the best values used in the paper.
    """

    def __init__(self):
        self.model = None

    def init_model(self, input_size, n_hidden=50, n_tensors=20, n_layers=2,
                  dropout_rate=0.5, rnn_type='gru',
                  optimizer='nadam', loss_function='categorical_crossentropy'):
        """
        Initialize model.

        Parameters
        ----------
        input_size: int
            Input size for this model.

        n_hidden: int, optional, default: 50
            Number of hidden units.

        n_tensors: int, optional, default: 20
            Number of tensors that capture the compositions between input tokens
            and the prototypes (aspect and opinion).

        n_layers: int, optional, default: 2
            Number of layers for the attention network.

        dropout_rate: float, optional, default: 0.5
            Dropout rate to used during training.

        rnn_type: {'gru', 'lstm', 'bilstm', 'bigru'}, optional, default: 'gru'
            Type of RNN used.

        optimizer: tf.keras.optimizers, optional, default: 'nadam'
            Optimizer to use during training.
            See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers.

        loss_function: tf.keras.losses, optional, default: 'categorical_crossentropy'
            Loss function to use during training.
            See https://www.tensorflow.org/api_docs/python/tf/keras/losses.
        """
        input = layers.Input(shape=(None, input_size))

        if rnn_type == 'gru':
            rnn = layers.GRU(units=n_hidden,
                             recurrent_activation='sigmoid',
                             return_sequences=True,
                             kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                             recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                             dropout=dropout_rate)(input)

        elif rnn_type == 'lstm':
            rnn = layers.LSTM(units=n_hidden,
                             recurrent_activation='sigmoid',
                             return_sequences=True,
                             kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                             recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                             dropout=dropout_rate)(input)

        elif rnn_type == 'bilstm':
            rnn = layers.Bidirectional(layers.LSTM(units=n_hidden,
                                                   recurrent_activation='sigmoid',
                                                   return_sequences=True,
                                                   kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                   recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                   dropout=dropout_rate))(input)

        elif rnn_type == 'bigru':
            rnn = layers.Bidirectional(layers.GRU(units=n_hidden,
                                                  recurrent_activation='sigmoid',
                                                  return_sequences=True,
                                                  kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                  recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                  dropout=dropout_rate))(input)

        else:
            raise ValueError("Unknown rnn type. Valid types are gru, lstm, bilstm, or bigru.")

        dropout = layers.Dropout(dropout_rate)(rnn)

        if rnn_type == 'gru' or rnn_type == 'lstm':
            cmla = CMLA(n_tensors, n_hidden, n_layers, dropout_rate, rnn_type)(dropout)

        elif rnn_type == 'bilstm' or rnn_type == 'bigru':
            cmla = CMLA(n_tensors, 2 * n_hidden, n_layers, dropout_rate, rnn_type)(dropout)

        aspect_prob = layers.Dense(3, activation='softmax',
                                   kernel_initializer=initializers.random_uniform(-0.2, 0.2))(cmla[0])

        opinion_prob = layers.Dense(3, activation='softmax',
                                    kernel_initializer=initializers.random_uniform(-0.2, 0.2))(cmla[1])

        self.model = tf.keras.Model(inputs=input, outputs=[aspect_prob, opinion_prob])
        self.model.compile(optimizer=optimizer,
                       loss=loss_function,
                       metrics=['accuracy'])

    def get_summary(self):
        """
        Get model summary.
        """
        if self.model != None:
            return self.model.summary()
        else:
            return None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            model_filename=None, epoch=1, batch_size=1, verbose=0, patience=1):
        """
        Fit model on data.

        Parameters
        ----------
        X_train: Train data.
        y_train: Train data labels.
        X_val: Validation data.
        y_val: Validation data labels.
        model_filename: filename for the model (for checkpoint).
        epoch: Number of epochs.
        batch_size: Size of train batch.
        verbose: Verbosity.
        patience: Patience value for early stopping.
        """
        # mini-batch
        if batch_size > 1:
            if X_val is not None:
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=patience)
                mc = ModelCheckpoint(model_filename, monitor='val_loss', mode='min', verbose=verbose,
                                      save_best_only=True)
                self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                               batch_size=batch_size, epochs=epoch, verbose=verbose, callbacks=[es, mc])
            else:
                self.model.fit(X_train, y_train,
                               batch_size=batch_size, epochs=epoch, verbose=verbose)

        # incremental learning
        else:
            if X_val is not None:
                min_loss = float('inf')
                patience_count = 0
                for i in range(epoch):
                    # shuffle data on each epoch
                    p = np.random.permutation(len(X_train))
                    for j in p:
                        self.model.fit(X_train[j], y_train[j], batch_size=1, verbose=verbose)

                    epoch_loss = self.get_validation_loss(X_val, y_val, batch_size, verbose)
                    print('Done with epoch', i, ', epoch error =', epoch_loss, ', min error =', min_loss)

                    if epoch_loss < min_loss:
                        min_loss = epoch_loss
                        patience_count = 0
                        self.save(model_filename)
                    else:
                        patience_count += 1

                    if patience_count == patience:
                        print('Epoch', epoch, 'early stopping')
                        break
            else:
                for i in range(epoch):
                    # shuffle data on each epoch
                    p = np.random.permutation(len(X_train))
                    for j in p:
                        self.model.fit(X_train[j], y_train[j], batch_size=1, verbose=verbose)

    def get_validation_loss(self, X_val, y_val, batch_size=1, verbose=0):
        """
        Get validation loss value.

        Parameters
        ----------
        X_val: Validation data.
        y_val: Validation data labels.
        batch_size: Size of train batch.
        verbose: Verbosity.

        Returns
        -------
        validation loss.
        """
        # mini-batch
        if batch_size > 1:
            return self.model.evaluate(X_val, y_val, batch_size=batch_size, verbose=verbose)[0]

        # incremental learning
        else:
            loss = 0
            for i in range(len(X_val)):
                loss += self.model.evaluate(X_val[i], y_val[i], batch_size=1, verbose=verbose)[0]
            return loss

    def predict(self, X):
        """
        Predict the labels for the given data.

        Parameters
        ----------
        X: Data to predict.

        Returns
        -------
        Labels for each tokens in data (B-ASPECT, I-ASPECT, B-SENTIMENT, I-SENTIMENT, or O).
        """
        y = []
        for i in range(len(X)):
            ya_score, yo_score = self.model.predict(X[i])

            # Get the label index with the highest probability
            ya_pred = np.argmax(ya_score, 2)
            yo_pred = np.argmax(yo_score, 2)

            y_pred = []
            for j in range(len(ya_pred[0])):
                if ya_pred[0][j] == 0:
                    # Both results are O
                    if yo_pred[0][j] == 0:
                        y_pred.append('O')

                    # Aspect prediction is O and opinion is not O
                    else:
                        if yo_pred[0][j] == 1:
                            y_pred.append('B-SENTIMENT')
                        else:
                            y_pred.append('I-SENTIMENT')

                elif yo_pred[0][j] == 0:
                    # Aspect prediction is not O and opinion is O
                    if ya_pred[0][j] == 1:
                        y_pred.append('B-ASPECT')
                    else:
                        y_pred.append('I-ASPECT')

                # Both results are not O
                else:
                    if ya_score[0][j][ya_pred[0][j]] >= yo_score[0][j][yo_pred[0][j]]:
                        if ya_pred[0][j] == 1:
                            y_pred.append('B-ASPECT')
                        else:
                            y_pred.append('I-ASPECT')
                    else:
                        if yo_pred[0][j] == 1:
                            y_pred.append('B-SENTIMENT')
                        else:
                            y_pred.append('I-SENTIMENT')

            y.append(y_pred)
        return y

    def evaluate(self, X, y, sentences=None):
        """
        Evaluate the model using the given data.

        Parameters
        ----------
        X: Evaluation data.
        y: Evaluation data labels.
        sentences: Evaluation data sentences, used to print the wrong results.

        Returns
        -------
        Evaluation score (precision, recall, and f1-score)
        """
        y_pred = self.predict(X)
        y_true = []
        y_pred2 = []

        for seq in y:
            for label in seq:
                if label == 'O':
                    y_true.append(0)
                elif label == 'B-ASPECT':
                    y_true.append(1)
                elif label == 'I-ASPECT':
                    y_true.append(2)
                elif label == 'B-SENTIMENT':
                    y_true.append(3)
                elif label == 'I-SENTIMENT':
                    y_true.append(4)

        for seq in y_pred:
            for label in seq:
                if label == 'O':
                    y_pred2.append(0)
                elif label == 'B-ASPECT':
                    y_pred2.append(1)
                elif label == 'I-ASPECT':
                    y_pred2.append(2)
                elif label == 'B-SENTIMENT':
                    y_pred2.append(3)
                elif label == 'I-SENTIMENT':
                    y_pred2.append(4)

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred2))
        print()
        print("Precision:")
        print(precision_score(y_true, y_pred2, average=None))
        print()
        print("Recall:")
        print(recall_score(y_true, y_pred2, average=None))
        print()
        print("F1-score:")
        print(f1_score(y_true, y_pred2, average=None))
        print()
        print("Report (entity level):")
        print(classification_report(y, y_pred))
        print(performance_measure(y, y_pred))

        if sentences != None:
            self.get_wrong_predictions(y, y_pred, sentences)

    def get_wrong_predictions(self, y, y_pred, sentences):
        """
        Print the wrong predictions for the given data.

        Parameters
        ----------
        y: real labels.
        y_pred: predicted labels.
        sentences: sentences that were being predicted.
        """
        count = 0
        for idx in range(len(y)):
            wrong = False
            for idx2 in range(len(y[idx])):
                if y[idx][idx2] != y_pred[idx][idx2]:
                    if not(wrong):
                        print("")
                        print('sentence:', " ".join(sentences[idx]))
                        print('labels:', " ".join(y[idx]))
                    wrong = True
                    count += 1
                    print(sentences[idx][idx2], '\t| P:', y_pred[idx][idx2], '\t| A:', y[idx][idx2])
        print(count, 'words misclasified')

    def save(self, model_filename):
        """
        Save model.

        Parameters
        ----------
        model_filename: filename for the model.
        """
        self.model.save(model_filename)

    @classmethod
    def load(cls, model_filename):
        """
        Load model.

        Parameters
        ----------
        model_filename: filename for the model.
        """
        self = cls()
        custom_objects={'CMLA': CMLA}
        self.model = load_model(model_filename, custom_objects = custom_objects)
        return self
