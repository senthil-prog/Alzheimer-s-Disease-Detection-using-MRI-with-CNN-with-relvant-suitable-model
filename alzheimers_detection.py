import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
from preprocess_data import create_data_generators

# Paths
train_dir = "data/train"
test_dir = "data/test"
model_path = "models/alzheimers_cnn_model.h5"
logs_path = "results/training_logs.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

# -------------------------
# Load data generators
# -------------------------
train_gen, val_gen, test_gen = create_data_generators(train_dir, test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

# -------------------------
# Build CNN Model
# -------------------------
def build_cnn(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification: AD or CN
    ])
    return model

model = build_cnn()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------
# Callbacks
# -------------------------
checkpoint_cb = callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max')
csv_logger = callbacks.CSVLogger(logs_path)
early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy')

# -------------------------
# Train Model
# -------------------------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint_cb, csv_logger, early_stop]
)

# -------------------------
# Save final model
# -------------------------
model.save(model_path)
print(f"✅ Model saved at: {model_path}")

# -------------------------
# Evaluate on Test Set
# -------------------------
loss, acc = model.evaluate(test_gen)
print(f"📊 Test Accuracy: {acc*100:.2f}%")

# -------------------------
# Plot training curves
# -------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("results/training_accuracy.png")

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("results/training_loss.png")

print("📈 Training plots saved to 'results/' directory.")
