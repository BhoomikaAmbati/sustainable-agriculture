import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

# Dataset path
data_dir = os.path.join("datasets", "plantvillage")

# Parameters
img_height = 180
img_width = 180
batch_size = 32

# Create training dataset
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Create validation dataset
val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Save class names BEFORE prefetch
class_names = train_ds.class_names

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print(" Preprocessing complete!")
print("Classes found:", class_names)
print("Number of training batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Number of validation batches:", tf.data.experimental.cardinality(val_ds).numpy())