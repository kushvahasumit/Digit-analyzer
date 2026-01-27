import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == '__main__':
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Ensure correct shape
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)

    # Build model
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='sgd',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['sparse_categorical_accuracy']
    )

    # Create checkpoint directory if it doesn't exist
    os.makedirs('checkpoint', exist_ok=True)
    checkpoint_save_path = "checkpoint/Regression.weights.h5"
    
    if os.path.exists(checkpoint_save_path + '.index'):
        print("Loading existing weights...")
        model.load_weights(checkpoint_save_path)

    # Checkpoint callback
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_sparse_categorical_accuracy',
        verbose=1
    )

    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=20,
        validation_data=(x_test, y_test),
        validation_freq=1,
        callbacks=[cp_callback],
        verbose=1
    )

    # Create models directory if it doesn't exist
    os.makedirs('app/models', exist_ok=True)
    
    # Save model
    model.save('app/models/regression.h5')
    print("Model saved to app/models/regression.h5")

    # Print model summary
    model.summary()

    # Plot training history
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('regression_training.png')
    print("Training plot saved as regression_training.png")
    plt.show()