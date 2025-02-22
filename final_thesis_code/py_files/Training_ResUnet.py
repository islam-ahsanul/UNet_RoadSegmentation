import os
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate,
    Activation, Conv2DTranspose, Input, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

# Seeding for reproducibility
os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
batch_size = 8
lr = 1e-4
epochs = 100
height = 512
width = 768
num_classes = 3  # "My Way", "Other Way", "Non-Drivable Area"

# Paths
dataset_path = "/path/to/processed_dataset/aug"
files_dir = "/path/to/files/aug"
model_file = os.path.join(files_dir, "resunet-multiclass.keras")
log_file = os.path.join(files_dir, "log-resunet.csv")

# Ensure directories exist
os.makedirs(files_dir, exist_ok=True)
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

# Residual Block
def residual_block(inputs, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)

    # Residual connection
    shortcut = Conv2D(num_filters, kernel_size=1, padding="same")(inputs)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = residual_block(inputs, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = concatenate([x, skip])
    x = residual_block(x, num_filters)
    return x

def build_resunet(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = residual_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)
    model = Model(inputs, outputs, name="ResU-Net")
    return model

# Dataset Loading
def load_data(path):
    train_x = sorted(glob(os.path.join(path, "train", "images", "*.png")))
    train_y = sorted(glob(os.path.join(path, "train", "masks", "*.png")))
    valid_x = sorted(glob(os.path.join(path, "valid", "images", "*.png")))
    valid_y = sorted(glob(os.path.join(path, "valid", "masks", "*.png")))
    return (train_x, train_y), (valid_x, valid_y)

# Preprocessing Functions
def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (width, height))  # Ensure the correct image size
    x = x / 255.0
    return x.astype(np.float32)

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read mask as grayscale
    x = cv2.resize(x, (width, height), interpolation=cv2.INTER_NEAREST)

    unique_values = np.unique(x)
    print(f"Unique mask values: {unique_values}")  # Debugging step

    if not np.all(np.isin(unique_values, [0, 128, 255])):  # Check grayscale values
        raise ValueError(f"Unexpected values found in mask: {unique_values}")

    x = np.where(x == 128, 1, x)  # Convert 128 to class 1
    x = np.where(x == 255, 2, x)  # Convert 255 to class 2
    x = np.clip(x, 0, num_classes - 1).astype(np.int32)  
    x = tf.one_hot(x, num_classes)  # Convert to one-hot encoding for training
    return x.numpy().astype(np.float32)

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([height, width, 3])
    y.set_shape([height, width, num_classes])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# IoU Metric
def iou_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + tf.keras.backend.epsilon())

# Load dataset
(train_x, train_y), (valid_x, valid_y) = load_data(dataset_path)
train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

# Build model
input_shape = (height, width, 3)
model = build_resunet(input_shape, num_classes)

# Compile model with fixed learning rate
model.compile(
    loss="categorical_crossentropy",  # Use categorical crossentropy for one-hot labels
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=["accuracy", iou_metric]
)

# Callbacks with the correct file extension
callbacks = [
    ModelCheckpoint(model_file, verbose=1, save_best_only=False, save_freq='epoch'),
    CSVLogger(log_file),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, min_lr=1e-7),  # Keep adaptive learning rate
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
]

# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    callbacks=callbacks
)

# Manual saving in case ModelCheckpoint fails
model.save(model_file)
print("Training complete. Model saved at:", model_file)
