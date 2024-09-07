
# MNIST Classification with Keras

## Introduction

This project aims to classify handwritten digits from the MNIST dataset using a Multi-Layer Perceptron (MLP) model. The MNIST dataset consists of 70,000 images (60,000 for training and 10,000 for testing) of handwritten digits (0-9), where each image is 28x28 pixels. The tasks involved in this project are:

1. Load the MNIST dataset and display the size of the training and test sets.
2. Build a simple MLP model with one hidden layer and report its accuracy.
3. Experiment with different numbers of hidden layers and observe the effect on accuracy.
4. Experiment with different sizes for the hidden layer and observe the effect on accuracy.
5. Analyze the findings from the above experiments.

## Dataset

The MNIST dataset is available through the Keras library's `datasets` module. It contains pre-split training and test datasets, making it easy to load and use.

For more details, refer to the official [MNIST Dataset Documentation](https://keras.io/api/datasets/mnist/).

## Dependencies

The project uses the following Python libraries:
- TensorFlow/Keras for building the neural network.
- Numpy for data manipulation.
- Matplotlib for visualization (optional).

Install these dependencies using pip:

```bash
pip install tensorflow numpy matplotlib
```

## Code Implementation

### 1. Load the MNIST Dataset and Display Dataset Sizes

The first step is to load the MNIST dataset using Keras and display the sizes of the training and test datasets.

```python
import numpy as np
from keras.datasets import mnist

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Display dataset sizes
print(f"Training set size: {train_images.shape[0]} samples")
print(f"Test set size: {test_images.shape[0]} samples")
```

Output:
```
Training set size: 60000 samples
Test set size: 10000 samples
```

### 2. Build a Simple MLP Model and Report Accuracy

In this step, we build an MLP model with one hidden layer of 512 neurons and use the `relu` activation function. The output layer has 10 neurons corresponding to the 10 digit classes (0-9) and uses the `softmax` activation function.

```python
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# Preprocess the dataset
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the MLP model
model = Sequential([
    Flatten(input_shape=(28 * 28,)),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
```

### 3. Experiment with Different Numbers of Hidden Layers

We experiment with MLP architectures by varying the number of hidden layers, setting the number of layers to [2, 4, 6, 8, 10] with each hidden layer containing 100 neurons. The model architecture is built dynamically based on the number of hidden layers.

```python
# Function to build and evaluate MLP model with different hidden layers
def build_and_evaluate_model(num_hidden_layers):
    model = Sequential([Flatten(input_shape=(28*28,))])
    
    # Add hidden layers dynamically
    for _ in range(num_hidden_layers):
        model.add(Dense(100, activation='relu'))
    
    model.add(Dense(10, activation='softmax'))
    
    # Compile and train the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)
    
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test accuracy with {num_hidden_layers} hidden layers: {test_acc}")

# Experiment with different numbers of hidden layers
for num_hidden_layers in [2, 4, 6, 8, 10]:
    build_and_evaluate_model(num_hidden_layers)
```

### 4. Experiment with Different Hidden Layer Sizes

In this step, we vary the size of the hidden layers while keeping the number of layers constant at one. The sizes used are [50, 100, 150, 200].

```python
# Function to build and evaluate MLP model with different hidden layer sizes
def build_and_evaluate_model_with_size(hidden_layer_size):
    model = Sequential([Flatten(input_shape=(28*28,))])
    model.add(Dense(hidden_layer_size, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    # Compile and train the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)
    
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test accuracy with hidden layer size {hidden_layer_size}: {test_acc}")

# Experiment with different hidden layer sizes
for hidden_layer_size in [50, 100, 150, 200]:
    build_and_evaluate_model_with_size(hidden_layer_size)
```

### 5. Key Findings

Based on the experiments with different numbers of hidden layers and varying hidden layer sizes, the following findings were observed:

1. **Increasing the number of hidden layers** improves model performance up to a point. However, adding too many layers can cause overfitting or result in diminishing returns.
2. **Increasing the size of the hidden layer** improves the model's ability to capture complex features but may also lead to overfitting if the size is too large relative to the dataset.
3. There is a trade-off between model complexity and accuracy. Simpler models with fewer layers and smaller sizes can perform reasonably well while being more computationally efficient.

## Conclusion

This project demonstrates how to build and experiment with different architectures of MLP models using the MNIST dataset. By varying the number of hidden layers and layer sizes, we can observe the trade-offs between model complexity and accuracy, helping to optimize the performance for specific tasks.

## References

- MNIST Dataset: [https://keras.io/api/datasets/mnist/](https://keras.io/api/datasets/mnist/)
- Keras Documentation: [https://keras.io/](https://keras.io/)
