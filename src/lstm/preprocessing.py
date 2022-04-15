from sklearn.preprocessing import MinMaxScaler
import numpy as np
SMOOTHING_WINDOW_SIZE = 365


def convertToClassification(x_val, y_val, step):
  """
  Converts timeseries of stock prices into classification up/down prediction
  classUp = 1
  classDown = 0
  """
  Y = y_val > x_val[:-step]
  y_val = Y.astype(int)

  return y_val


def normalizeAtOnce(data):
  scaler = MinMaxScaler()
  scaler.fit(data)
  return scaler.transform(data)


def normalize(data):
  scaler = MinMaxScaler()
  # Train the Scaler with training data and smooth data
  for di in range(0, len(data)-1, SMOOTHING_WINDOW_SIZE):
    scaler.fit(data[di:di+SMOOTHING_WINDOW_SIZE, :])
    data[di:di+SMOOTHING_WINDOW_SIZE,
         :] = scaler.transform(data[di:di+SMOOTHING_WINDOW_SIZE, :])

  # You normalize the last bit of remaining data
  data[di+SMOOTHING_WINDOW_SIZE:,
       :] = scaler.transform(data[di+SMOOTHING_WINDOW_SIZE:, :])
  return data
