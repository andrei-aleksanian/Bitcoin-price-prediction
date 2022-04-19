import tensorflow as tf

path_checkpoint = "checkpoints/model_checkpoint.h5"
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)
