import pickle

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error


class SalaryPredictor:
    def __init__(self):
        self.model = None

    def build(self, X_categorical, X_continuous):
        cat_inputs = []
        cat_embeds = []
        for i, column_name in enumerate(X_categorical.columns):
            cat_input = Input(shape=(1,), name=f'cat_input_{i}')
            cat_embed = Embedding(X_categorical[column_name].nunique(), 15,
                                  name=f'cat_embedding_{i}')(cat_input)
            cat_embed = Flatten()(cat_embed)
            cat_inputs.append(cat_input)
            cat_embeds.append(cat_embed)

        # concatenate all categorical embeddings
        cat_concat = Concatenate(name='cat_concat')(cat_embeds)

        # dense layer for continuous features
        cont_input = Input(shape=(X_continuous.shape[1],), name='cont_input')
        cont_dense = Dense(16, activation='relu', name='cont_dense')(cont_input)

        # concatenate categorical embedding and continuous dense layers
        concat = Concatenate(name='concat')([cat_concat, cont_dense])

        # output layer for regression
        regression = Dense(16, activation='linear', name='regression')(concat)
        output = Dense(1, activation='linear', name='output')(regression)

        # define and compile the model
        model = tf.keras.models.Model(inputs=cat_inputs + [cont_input], outputs=output)
        model.compile(optimizer='adam', loss='mse')

        self.model = model

    def train(self, X_categorical, X_continuous, y, batch_size=4, epochs=100, validation_split=0.2,
              checkpoint_path='best_model.h5'):
        # define model checkpoints and early stopping
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

        # train the model
        self.model.fit(x=[X_categorical[col].values for col in X_categorical.columns] + [X_continuous.values],
                       y=y.values,
                       batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                       callbacks=[checkpoint, early_stop])

    def load(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, X_categorical, X_continuous):
        return self.model.predict([X_categorical[col].values for col in X_categorical.columns] + [X_continuous.values])

    def test(self, X_categorical, X_continuous, y_true, list_errors=False):
        y_pred = self.predict(X_categorical, X_continuous)

        with open('scalers/salary_in_usd.pkl', 'rb') as f:
            scaler = pickle.load(f)
        y_true = scaler.inverse_transform(y_true.values.reshape(-1, 1)).reshape(-1)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
        mae = mean_absolute_error(y_true, y_pred)
        if list_errors:
            for true_sample, pred_sample in zip(y_true, y_pred):
                print(f"Predicted: {pred_sample:.2f}\tTrue: {true_sample:.2f}\n" + "=" * 10)
        return mae


if __name__ == '__main__':
    model = SalaryPredictor()
    with open(f'train_data/X_categorical.pkl', 'rb') as f:
        X_categorical = pickle.load(f)
    with open(f'train_data/X_continuous.pkl', 'rb') as f:
        X_continuous = pickle.load(f)
    with open(f'train_data/y.pkl', 'rb') as f:
        y = pickle.load(f)
    model.build(X_categorical, X_continuous)
    print(model.model.summary())
    model.train(X_categorical, X_continuous, y)

    with open(f'test_data/X_categorical.pkl', 'rb') as f:
        X_categorical = pickle.load(f)
    with open(f'test_data/X_continuous.pkl', 'rb') as f:
        X_continuous = pickle.load(f)
    with open(f'test_data/y.pkl', 'rb') as f:
        y = pickle.load(f)

    model.load("best_model.h5")
    print(model.test(X_categorical, X_continuous, y, list_errors=True))
