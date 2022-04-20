from os.path import exists
from tabnanny import verbose

import tensorflow as tf
import pandas as pd
import numpy as np


def evaluateFinal(getModel, dataset_train, dataset_val, config, name):
  """
  Train a model 10 times, record minimum RMSE and MSE, save in a csv.
  Record mean and standard deviation.
  """

  if exists("results/{name}.csv"):
    raise FileExistsError("Please, name the model something else")

  results = np.empty((0, 2))
  for batch in dataset_train.take(1):
    inputs, _ = batch

  print("Training Started...")
  print("Iterations:")

  # Train the model 10 times and record best validation RMSE:
  for i in range(10):
    model = getModel(inputs, config["future"])
    history = model.fit(
        dataset_train,
        epochs=config["epochs"],
        validation_data=dataset_val,
        verbose=0
    )
    mse = history.history["val_loss"]
    rmse = history.history["val_rmse"]
    results = np.vstack((results, np.array([min(mse), min(rmse)]).reshape(-1)))
    print(i+1)

  df = pd.DataFrame(results, columns=["MSE", "RMSE"])

  # Gather mean and standard deviation of RMSE
  mean = np.mean(results[1])
  std = np.std(results[1])
  df2 = pd.DataFrame([[mean, std]], columns=["RMSE mean", "RMSE std"])
  all = pd.concat([df, df2], axis=1)

  # Record results in a csv file
  all.to_csv(f"results/{name}.csv")
  print("Done")


def evaluateModelQuick(model, x_test, y_test):
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
