import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
import warnings
warnings.filterwarnings("ignore")

#Import the preprocessed metadata and set image directories
METADATA_PATH = r"C:/Users/adity/Documents/Semester 8/Healthcare Analytics Theory/preprocessed_metadata.csv"
IMG_DIR_1 = r"C:/Users/adity/Documents/Semester 8/Healthcare Analytics Theory/Datasets/HAM10000_images_part_1"
IMG_DIR_2 = r"C:/Users/adity/Documents/Semester 8/Healthcare Analytics Theory/Datasets/HAM10000_images_part_2"


df = pd.read_csv(METADATA_PATH)



df["image_path"] = df["image_id"].apply(resolve_image_path)

# Remove rows without images (safety)
df = df.dropna(subset=["image_path"])

print("Total usable samples:", len(df))

# TRAIN / VALIDATION SPLIT
train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

# IMAGE GENERATOR CONFIG
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.10,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

# GENERATORS
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="dx",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="dx",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nIMAGE PRE-PROCESSING COMPLETED SUCCESSFULLY\n")

# EXPORT DATAFRAMES FOR REUSE
__all__ = ["train_generator", "val_generator", "train_df", "val_df"]
