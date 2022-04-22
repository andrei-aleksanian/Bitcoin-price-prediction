import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

SMOOTHING_WINDOW_SIZE = 1500


def splitTrainValTest(normalized_data, split_train_val_fraction, split_val_test_fraction):
  train_val_split = int(split_train_val_fraction *
                        int(normalized_data.shape[0]))
  val_test_split = int(split_val_test_fraction * int(normalized_data.shape[0]))

  train_data = normalized_data[:train_val_split]
  val_data = normalized_data[train_val_split:val_test_split]
  test_data = normalized_data[val_test_split:]
  return train_data, val_data, test_data, train_val_split, val_test_split


def getDatasetsMultiDimensional(config, normalized_data):
  past = config["past"]
  future = config["future"]
  batch_size = config["batch_size"]

  train_data, val_data, test_data, train_val_split, val_test_split = splitTrainValTest(
      normalized_data, 0.8, 0.9)

  # Train data
  x_train = train_data
  y_train = normalized_data[past: future+train_val_split, 3]

  dataset_train = tf.keras.preprocessing.timeseries_dataset_from_array(
      x_train,
      y_train,
      sequence_length=past,
      batch_size=batch_size,
  )

  # Validation data
  x_val = val_data
  y_val = normalized_data[train_val_split+past:future+val_test_split, 3]

  dataset_val = tf.keras.preprocessing.timeseries_dataset_from_array(
      x_val,
      y_val,
      sequence_length=past,
      batch_size=batch_size,
  )

  # Test data - used in a manual testing dataset
  x_test = test_data
  y_test = normalized_data[val_test_split + past:]

  return dataset_train, dataset_val, x_test, y_test


def getDatasets(config, normalized_data):
  past = config["past"]
  future = config["future"]
  batch_size = config["batch_size"]

  train_data, val_data, test_data, train_val_split, val_test_split = splitTrainValTest(
      normalized_data, 0.8, 0.9)

  # Train data
  x_train = train_data
  y_train = normalized_data[past: future+train_val_split]

  dataset_train = tf.keras.preprocessing.timeseries_dataset_from_array(
      x_train,
      y_train,
      sequence_length=past,
      batch_size=batch_size,
  )

  # Validation data
  x_val = val_data
  y_val = normalized_data[train_val_split+past:future+val_test_split]

  dataset_val = tf.keras.preprocessing.timeseries_dataset_from_array(
      x_val,
      y_val,
      sequence_length=past,
      batch_size=batch_size,
  )

  # Test data - used in a manual testing dataset
  x_test = test_data
  y_test = normalized_data[val_test_split + past:]

  return dataset_train, dataset_val, x_test, y_test


def prepareTestSetMultiDimensional(x, y, past, future):
  """Split x and y into batches of 'past' x that predict 'future' y."""
  x_split = np.empty((0, past, x.shape[1]))
  y_split = np.empty((0, future))
  for i in range(x.shape[0]-past-future):
    x_split = np.vstack((x_split, np.array(x[i:past+i]).reshape(1, 10, 6)))
    y_split = np.vstack((y_split, np.array(y[i:i+future]).reshape(-1)))
  return x_split, y_split


def prepareTestSet(x, y, past, future):
  """Split x and y into batches of 'past' x that predict 'future' y."""
  x_split = np.empty((0, past))
  y_split = np.empty((0, future))
  for i in range(x.shape[0]-past-future):
    x_split = np.vstack((x_split, np.array(x[i:past+i]).reshape(-1)))
    y_split = np.vstack((y_split, np.array(y[i:i+future]).reshape(-1)))
  return x_split, y_split


def normalizeAtOnce(data):
  scaler = MinMaxScaler()
  scaler.fit(data)
  return scaler.transform(data), scaler


def normalize(data):
  scaler = MinMaxScaler()
  # Train the Scaler with training data and smooth data
  for di in range(0, len(data)-1, SMOOTHING_WINDOW_SIZE):
    scaler.fit(data[di:di+SMOOTHING_WINDOW_SIZE, :])
    data[di:di+SMOOTHING_WINDOW_SIZE,
         :] = scaler.transform(data[di:di+SMOOTHING_WINDOW_SIZE, :])

  # You normalize the last bit of remaining data
  scaler.fit(data[di:, :])
  data[di:, :] = scaler.transform(data[di:, :])
  return data, scaler
