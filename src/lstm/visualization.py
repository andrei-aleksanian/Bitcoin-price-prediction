import matplotlib.pyplot as plt
import numpy as np


def show_semilogy(date, data):
  plt.semilogy(date, data)
  plt.xlabel('Date')
  plt.ylabel('Close Price')
  plt.show()


def showRegressionExampleMultiDimensional(x, y, model, scaler=None, withNews=False):
  y_hat = model(x.reshape(1, x.shape[0], x.shape[1]))
  y_hat = y_hat.numpy().reshape(-1)

  if not scaler is None:
    dummyDimensions = (x.shape[0], x.shape[1] - 3) if withNews else x.shape
    y_hat_dummy = np.zeros(dummyDimensions)
    y_hat_dummy[:, 3] = y_hat
    y_dummy = np.zeros(dummyDimensions)
    y_dummy[:, 3] = y

    y_hat_dummy = scaler.inverse_transform(y_hat_dummy)
    y_dummy = scaler.inverse_transform(y_dummy)

    y_hat = y_hat_dummy[:, 3]
    y = y_dummy[:, 3]

  plt.plot(range(y.shape[0]), y)
  plt.plot(range(y_hat.shape[0]), y_hat)
  plt.xlabel('Date')
  plt.ylabel('Close Price')
  plt.show()


def showRegressionExample(x, y, model, scaler=None):
  y_hat = model(x.reshape(1, -1))
  y_hat = y_hat.numpy().reshape(-1)

  if not scaler is None:
    y_hat = scaler.inverse_transform(y_hat.reshape(-1, 1)).reshape(-1)
    y = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)

  plt.plot(range(y.shape[0]), y)
  plt.plot(range(y_hat.shape[0]), y_hat)
  plt.xlabel('Date')
  plt.ylabel('Close Price')
  plt.show()


def show_batch(x, y, past):
  plt.plot(range(x.shape[0]), x)
  plt.plot(range(past, past+y.shape[0]), y)
  plt.xlabel('Date')
  plt.ylabel('Close Price')
  plt.show()


def show_data_simple(data, date):
  plt.plot(range(data.shape[0]), data)
  plt.xticks(range(0, date.shape[0], 100), date.loc[::100][:10], rotation=45)
  plt.xlabel('Date')
  plt.ylabel('Close Price')
  plt.show()


def show_data(df):
  """Old, unused"""
  plt.figure(figsize=(18, 9))
  plt.plot(range(df.shape[0]), df['Close'])
  plt.xticks(range(0, df.shape[0], 200), df['Date'].loc[::200], rotation=45)
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Close Price', fontsize=18)
  plt.show()


def show_prediction(x, y, future):
  plt.figure(figsize=(18, 9))
  plt.plot(range(x.shape[0]), x, color='b', label='True')
  plt.plot(range(x.shape[0]), np.concatenate(
      [x[:-future], y]), color='orange', label='Prediction')
  plt.xlabel('Date')
  plt.ylabel('Close Price')
  plt.legend(fontsize=18)
  plt.show()


def visualize_rmse(history):
  loss = history.history["rmse"]
  val_loss = history.history["val_rmse"]
  epochs = range(len(loss))
  plt.figure()
  plt.plot(epochs, loss, "b", label="Training RMSE")
  plt.plot(epochs, val_loss, "r", label="Validation RMSE")
  plt.title("Training and Validation RMSE")
  plt.xlabel("Epochs")
  plt.ylabel("Error")
  plt.legend()
  plt.show()


def visualize_loss(history):
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  epochs = range(len(loss))
  plt.figure()
  plt.plot(epochs, loss, "b", label="Training loss")
  plt.plot(epochs, val_loss, "r", label="Validation loss")
  plt.title("Training and Validation Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()


def visualize_accuracy(history):
  plt.figure()
  plt.plot(history.history['acc'], 'r', label='train')
  plt.plot(history.history['val_acc'], 'b', label='val')
  plt.xlabel(r'Epoch', fontsize=20)
  plt.ylabel(r'Accuracy', fontsize=20)
  plt.legend()
  plt.tick_params(labelsize=20)


def show_result(x, y, future, model):
  """Visualize single step prediction after x days into future days."""
  plt.plot(range(x[0].shape[0]), x[0], color='b', label='True')
  plt.plot(x[0].shape[0]+future, model.predict(x)[0][0], marker="o", markersize=10,
           markeredgecolor="red", markerfacecolor="red", label="Predicted y")
  plt.plot(x[0].shape[0]+future, y[0], marker="o", markersize=10,
           markeredgecolor="blue", markerfacecolor="blue", label="True y")
  plt.xlabel('Date')
  plt.ylabel('Close Price')
  plt.legend()
  plt.show()
