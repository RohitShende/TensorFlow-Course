import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# getting the dataset from keras
fashion_mnist = keras.datasets.fashion_mnist

# load train and test images and labels
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# visualize one image from the train images using matplotlib
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# pre-processing and normalizing the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# defining a model with 3 layers
# layer 1 is input layer
# layer 2 is Dense layer with 128 neurons with relu activation
# layer 3 is Dense layer with 10 neurons and softmax as activation function for classification
model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(128, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)])

# compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(train_images, train_labels, epochs=5)

# evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

# predict
predictions = model.predict(test_images)
print(predictions[0])

# predicting class directly
print(np.argmax(predictions[0]))

# verify with test_label
print(test_labels[0])

assert np.argmax(predictions[0]) == test_labels[0]
print('Classified the Boot Correctly !')