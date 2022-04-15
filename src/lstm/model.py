import tensorflow as tf

LEARNING_RATE = 0.00001


def getModel(inputs):
  inputs = tf.keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
  lstm_out = tf.keras.layers.LSTM(10)(inputs)
  outputs = tf.keras.layers.Dense(2, activation="softmax")(lstm_out)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss="mse", metrics=['acc'])

  return model
