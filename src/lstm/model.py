import tensorflow as tf

LEARNING_RATE = 0.001


def getModel(inputs):
  inputs = tf.keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
  lstm_out = tf.keras.layers.LSTM(32)(inputs)
  outputs = tf.keras.layers.Dense(1)(lstm_out)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=tf.keras.optimizers.Adam(
      learning_rate=LEARNING_RATE), loss="mse", metrics=['acc'])

  return model
