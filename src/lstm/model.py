import tensorflow as tf
from keras.models import load_model


def evaluateModel(model, x_test, y_test):
  model.load_weights("checkpoints/model_checkpoint.h5")
  results = model.evaluate(x_test, y_test, batch_size=x_test.shape[0])

  print("---- TEST RESULTS ----")
  print(f"MSE loss - {results[0]}")
  print(f"RMSE - {results[1]}")


def getModel(inputs, outputLayer):
  inputs = tf.keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
  lstm_out = tf.keras.layers.LSTM(
      100, return_sequences=True, dropout=0.5)(inputs)
  lstm_out = tf.keras.layers.LSTM(100, dropout=0.5)(lstm_out)
  outputs = tf.keras.layers.Dense(outputLayer)(lstm_out)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

  return model
