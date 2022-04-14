import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

LEARNING_RATE = 0.001


def getModel(inputs):
  inputs = tf.keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
  lstm_out = tf.keras.layers.LSTM(32)(inputs)
  outputs = tf.keras.layers.Dense(1)(lstm_out)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=tf.keras.optimizers.Adam(
      learning_rate=LEARNING_RATE), loss="mse", metrics=['acc'])

  return model


def getWrappedModel(inputs):
  return KerasRegressor(build_fn=getModel, inputs=inputs)


def getGrid(inputs):
  model = getWrappedModel(inputs)
  batch_size = [10, 20, 40, 60, 80, 100]
  epochs = [10, 50, 100]
  param_grid = dict(batch_size=batch_size, epochs=epochs)
  grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
  return grid
