import numpy as np
from tensorflow.keras.applications import MobileNetV2 #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

# IMPORT GENERATORS FROM images_preprocessing.py
from images_preprocessing import train_generator, val_generator

NUM_CLASSES = 7
EPOCHS = 10
LEARNING_RATE = 1e-4

# BASE MODEL (TRANSFER LEARNING)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Freeze base layers

# CUSTOM CLASSIFIER
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# COMPILE MODEL
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# TRAIN MODEL
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

model.save("skin_cancer_multiclass_model.h5")
print("\nMODEL TRAINING COMPLETED & SAVED")
