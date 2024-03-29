from os.path import exists

import tensorflow as tf
import pandas as pd
import numpy as np
from lstm.callbacks import modelckpt_callback, es_callback, getCheckpointCb


def evaluateFinalBaseline(getModel, x_train, y_train, x_test, y_test, config, name):
  """
  Train a ANN model 10 times, record minimum RMSE, MAE and MAPE, save in a csv.
  Record mean and standard deviation.
  """
  if exists("results/{name}.csv"):
    raise FileExistsError("Please, name the model something else")

  results = np.empty((0, 3))

  print("Training Started...")
  print("Iterations:")

  # Train the model 10 times and record best validation RMSE:
  for i in range(10):
    model = getModel(config)
    model.fit(
        x=x_train,
        y=y_train,
        epochs=config["epochs"],
        validation_split=0.1,
        callbacks=[es_callback, getCheckpointCb(f"{name} {i}")],
        verbose=0
    )

    model.load_weights(f"temp/{name} {i}.h5")
    result = model.evaluate(x_test, y_test, batch_size=x_test.shape[0])
    results = np.vstack((results, result[1:]))
    print(i+1)

  df = pd.DataFrame(results, columns=["RMSE", "MAE", "MAPE"])

  # Gather mean and standard deviation of RMSE
  mean_rmse = np.mean(results[:, 0])
  std_rmse = np.std(results[:, 0])
  mean_mae = np.mean(results[:, 1])
  std_mae = np.std(results[:, 1])
  mean_mape = np.mean(results[:, 2])
  std_mape = np.std(results[:, 2])

  df2 = pd.DataFrame([[mean_rmse, std_rmse, mean_mae, std_mae,
                     mean_mape, std_mape]], columns=["RMSE mean", "RMSE std", "MAE mean", "MAE std", "MAPE mean", "MAPE std"])
  all = pd.concat([df, df2], axis=1)

  # Record results in a csv file
  all.to_csv(f"results/{name}.csv")
  print("Done")


def evaluateFinal(getModel, dataset_train, dataset_val, x_test, y_test, config, name):
  """
  Train a model 10 times, record minimum RMSE, MAE and MAPE, save in a csv.
  Record mean and standard deviation.
  """

  if exists("results/{name}.csv"):
    raise FileExistsError("Please, name the model something else")

  results = np.empty((0, 3))

  print("Training Started...")
  print("Iterations:")

  # Train the model 10 times and record best validation RMSE:
  for i in range(10):
    model = getModel(config)
    model.fit(
        dataset_train,
        epochs=config["epochs"],
        validation_data=dataset_val,
        verbose=0,
        callbacks=[es_callback, getCheckpointCb(f"{name} {i}")],
    )

    model.load_weights(f"temp/{name} {i}.h5")
    result = model.evaluate(x_test, y_test, batch_size=x_test.shape[0])

    # rmse = min(history.history["val_rmse"])
    # mae = min(history.history["val_mae"])
    # mape = min(history.history["val_mape"])
    results = np.vstack((results, result[1:]))
    print(i+1)

  df = pd.DataFrame(results, columns=["RMSE", "MAE", "MAPE"])

  # Gather mean and standard deviation of RMSE
  mean_rmse = np.mean(results[:, 0])
  std_rmse = np.std(results[:, 0])
  mean_mae = np.mean(results[:, 1])
  std_mae = np.std(results[:, 1])
  mean_mape = np.mean(results[:, 2])
  std_mape = np.std(results[:, 2])

  df2 = pd.DataFrame([[mean_rmse, std_rmse, mean_mae, std_mae,
                     mean_mape, std_mape]], columns=["RMSE mean", "RMSE std", "MAE mean", "MAE std", "MAPE mean", "MAPE std"])
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


def getModel(config):
  inputs = tf.keras.layers.Input(shape=(config["past"], config["features"]))
  lstm_out = tf.keras.layers.LSTM(
      config["neurons"], return_sequences=True, dropout=0.3)(inputs)
  lstm_out = tf.keras.layers.LSTM(config["neurons"], dropout=0.3)(lstm_out)
  outputs = tf.keras.layers.Dense(config["future"])(lstm_out)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss="mse",
      metrics=[
          tf.keras.metrics.RootMeanSquaredError(name="rmse"),
          tf.keras.metrics.MeanAbsoluteError(name="mae"),
          tf.keras.metrics.MeanAbsolutePercentageError(name="mape")
      ])

  return model


def getModelBaseline(config):
  inputs = tf.keras.layers.Input(
      shape=config["features"])
  dense = tf.keras.layers.Dense(config["future"])(inputs)
  dropout = tf.keras.layers.Dropout(0.3)(dense)
  outputs = tf.keras.layers.Dense(config["future"])(dropout)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss="mse",
      metrics=[
          tf.keras.metrics.RootMeanSquaredError(name="rmse"),
          tf.keras.metrics.MeanAbsoluteError(name="mae"),
          tf.keras.metrics.MeanAbsolutePercentageError(name="mape")
      ])

  return model
