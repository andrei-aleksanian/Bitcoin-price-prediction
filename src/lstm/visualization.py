import matplotlib.pyplot as plt


def visualize_loss(history, title):
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  epochs = range(len(loss))
  plt.figure()
  plt.plot(epochs, loss, "b", label="Training loss")
  plt.plot(epochs, val_loss, "r", label="Validation loss")
  plt.title(title)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()


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
