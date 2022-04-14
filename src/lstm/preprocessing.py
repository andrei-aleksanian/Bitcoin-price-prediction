from sklearn.preprocessing import MinMaxScaler

SMOOTHING_WINDOW_SIZE = 365


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
