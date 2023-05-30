###########################
# IMPORT REQUIRED LIBRARIES
###########################

import os
import warnings

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras

warnings.filterwarnings("ignore")

###########################
# LIMIT MEMORY GROWTH
###########################

# Retrieve a list of all available physical GPUs on the system
gpus = tf.config.experimental.list_physical_devices("GPU")

# Iterate over each GPU device
for gpu in gpus:
    # Set the memory growth option for the current GPU
    tf.config.experimental.set_memory_growth(gpu, True)

###########################
# RETRIEVE THE REQUIRED FILES
###########################

# Specify the directory path where the dataset is stored
dir = "data"

# Retrieve a list of file names within the "with_mask" directory
with_mask_files = os.listdir(os.path.join(dir, "with_mask"))

# Retrieve a list of file names within the "without_mask" directory
without_mask_files = os.listdir(os.path.join(dir, "without_mask"))

###########################
# CHECK IF THE DATASETS ARE BALANCED OR NOT
###########################

# Calculate the ratio between the lengths of the two lists
ratio = len(with_mask_files) / len(without_mask_files)

# Check if the ratio is greater than 70%
if ratio >= 0.7:
    print("The datasets are balanced.")
else:
    print("The datasets are imbalanced.")

###########################
# CREATE LABELS FOR THE TWO CLASS OF IMAGES
###########################

# with mask -> 1
# without mask -> 0

# Create a list of labels for the images with masks, assigning a value of 1 to each label
with_mask_labels = [1] * len(with_mask_files)

# Create a list of labels for the images without masks, assigning a value of 0 to each label
without_mask_labels = [0] * len(without_mask_files)

# Concatenate the two lists of labels to create a combined list of labels for all images
labels = with_mask_labels + without_mask_labels

###########################
# CHECK IF THE DIMENSIONS OF THE IMAGES ARE SAME OR NOT
###########################

# Specify the path to the first image file
img_path_1 = "data/with_mask/with_mask_2590.jpg"

# Read the first image file into an array
img_1 = mpimg.imread(img_path_1)

# Specify the path to the second image file
img_path_2 = "data/without_mask/without_mask_2925.jpg"

# Read the second image file into an array
img_2 = mpimg.imread(img_path_2)

# Check if the dimensions of the images are the same
if img_1.shape == img_2.shape:
    print("The dimensions of the images are the same.")
else:
    print("The dimensions of the images are different.")

###########################
# PREPROCESS THE IMAGES
###########################

# Initialize an empty list to store the image data
data = []

# Process images with masks and without masks
with_mask_path = "data/with_mask/"
without_mask_path = "data/without_mask/"

# Combine the two lists of image files
all_files = with_mask_files + without_mask_files

# Iterate over each image file
for img_file in all_files:
    # Determine the path based on whether it is a "with_mask" or "without_mask" image
    if img_file in with_mask_files:
        path = with_mask_path
    else:
        path = without_mask_path
    # Open the image file
    image = Image.open(path + img_file)
    # Resize the image to 128x128 pixels
    image = image.resize((128, 128))
    # Convert the image to RGB format if necessary
    image = image.convert("RGB")
    # Convert the image to a numpy array
    image = np.array(image)
    # Append the image array to the data list
    data.append(image)

###########################
# TRAIN-TEST SPLIT AND SCALING
###########################

# Convert the 'data' list to a numpy array
X = np.array(data)

# Convert the 'labels' list to a numpy array
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Scale the pixel values of the training and testing sets
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

###########################
# BUILD AND COMPILE CNN
###########################

# Define the number of classes
num_of_class = 2

# Initialize the Sequential model
model = Sequential()

# Add a 2D convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation
# Specify the input shape as (128, 128, 3) for images with size 128x128 and 3 color channels (RGB)
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(128, 128, 3)))

# Add a max pooling layer with a pool size of 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))


# Add another 2D convolutional layer with 64 filters, a 3x3 kernel, and ReLU activation
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))

# Add another max pooling layer with a pool size of 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))


# Flatten the output from the previous layer to a 1D array
model.add(Flatten())


# Add a dense layer with 128 units and ReLU activation
model.add(Dense(128, activation="relu"))

# Apply dropout regularization with a rate of 0.5
model.add(Dropout(0.5))


# Add another dense layer with 64 units and ReLU activation
model.add(Dense(64, activation="relu"))

# Apply dropout regularization with a rate of 0.5
model.add(Dropout(0.5))


# Add the final dense layer with 'num_of_class' units and sigmoid activation
# Sigmoid activation is used for binary classification
model.add(Dense(num_of_class, activation="sigmoid"))


# Compile the model with specified optimizer, loss function, and metrics
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

###########################
# TRAIN THE MODEL AND PLOT PERFORMANCE
###########################

# Train the model using the scaled training data and labels
hist = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=5)

# Create a new figure
fig = plt.figure()

# Plot the training loss curve in teal color and label it as "loss"
plt.plot(hist.history["loss"], color="teal", label="loss")

# Plot the validation loss curve in orange color and label it as "val_loss"
plt.plot(hist.history["val_loss"], color="orange", label="val_loss")

# Set the title of the figure as "Loss" with a font size of 20
fig.suptitle("Loss", fontsize=20)

# Add a legend to the plot at the upper right position
plt.legend(loc="upper right")

# Save the figure as "fig1.png"
plt.savefig("fig1.png")


# Create another new figure
fig = plt.figure()

# Plot the training accuracy curve in teal color and label it as "accuracy"
plt.plot(hist.history["accuracy"], color="teal", label="accuracy")

# Plot the validation accuracy curve in orange color and label it as "val_accuracy"
plt.plot(hist.history["val_accuracy"], color="orange", label="val_accuracy")

# Set the title of the figure as "Accuracy" with a font size of 20
fig.suptitle("Accuracy", fontsize=20)

# Add a legend to the plot at the upper left position
plt.legend(loc="upper left")

# Save the figure as "fig2.png"
plt.savefig("fig2.png")

###########################
# MODEL EVALUATION
###########################

# Evaluate the model using the scaled test data and labels
loss, accuracy = model.evaluate(X_test_scaled, y_test)

# Print the test accuracy
print("Test Accuracy: {} %".format(round(accuracy * 100, 3)))

###########################
# PREDICTIVE SYSTEM
###########################

# Prompt the user to input the path of the image to be predicted
input_image_path = input("Path of the image to be predicted: ")

# Read the image using OpenCV
input_image = cv2.imread(input_image_path)

# Display the input image
cv2.imshow("", input_image)

# Resize the input image to 128x128 pixels
input_image_resized = cv2.resize(input_image, (128, 128))

# Scale the pixel values of the resized image
input_image_scaled = input_image_resized / 255

# Reshape the scaled image to match the model's input shape
input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

# Make predictions on the reshaped image using the trained model
input_pred = model.predict(input_image_reshaped)

# Print the predicted probability distribution
print("Input Prediction: {}".format(input_pred))

# Get the label prediction by finding the index of the highest probability
input_pred_label = np.argmax(input_pred)

# Print the label prediction
print("Input Label Prediction: {}".format(input_pred_label))

# Check the label prediction and print a corresponding message
if input_pred_label == 1:
    print("The person in the image is wearing a mask")
else:
    print("The person in the image is not wearing a mask")
