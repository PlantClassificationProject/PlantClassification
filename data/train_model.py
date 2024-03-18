from tensorflow import keras

from generate_model import make_model

def train_model(model, image_size, train_ds, val_ds):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    epochs = 25
    callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")]

    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )