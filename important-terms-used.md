# Machine Learning Terms Glossary

## Core Frameworks

### TensorFlow
**Definition**: An open-source machine learning framework developed by Google.

**Why Used**: 
- Provides tools for building and training neural networks
- Handles complex mathematical operations automatically
- Supports both research and production deployment
- Works on CPUs, GPUs, and mobile devices

**Core Functionality**: Acts as the foundation that manages all computations, memory allocation, and optimization for your machine learning models.

---

### Keras
**Definition**: A high-level neural networks API that runs on top of TensorFlow.

**Why Used**:
- Simplifies neural network creation with user-friendly syntax
- Reduces code complexity compared to raw TensorFlow
- Provides pre-built layers and models
- Faster prototyping and development

**Core Functionality**: Acts as a simplified interface to TensorFlow, making it easier to build models without dealing with low-level details.

---

## Callbacks (Training Controllers)

### Checkpoint Callback
**Definition**: Automatically saves your model during training.

**Why Used**:
- Prevents losing progress if training crashes
- Saves only the best performing version
- Allows you to resume training from the best point

**Core Functionality**: Monitors validation accuracy and saves the model file whenever it improves, ensuring you keep the best version.

---

### Early Stop Callback
**Definition**: Automatically stops training when the model stops improving.

**Why Used**:
- Prevents overfitting (when model memorizes training data)
- Saves computational time and resources
- Finds the optimal stopping point automatically

**Core Functionality**: Watches validation loss and stops training if it doesn't improve for a specified number of epochs (patience).

---

### Reduce Learning Rate Callback
**Definition**: Automatically decreases the learning rate when training plateaus.

**Why Used**:
- Helps model fine-tune when it gets stuck
- Allows more precise weight updates
- Improves final model performance

**Core Functionality**: Monitors training progress and cuts learning rate in half when improvement stalls, allowing smaller, more precise adjustments.

---

## Neural Network Components

### Layers
**Definition**: Building blocks of neural networks that process and transform data.

**Why Used**:
- Each layer learns different features from the data
- Stacking layers creates complex pattern recognition
- Different layer types serve different purposes

**Core Functionality**: Transform input data through mathematical operations (matrix multiplication, activation functions) to extract meaningful features.

---

### MobileNetV2
**Definition**: A pre-trained convolutional neural network optimized for mobile devices.

**Why Used**:
- Already trained on millions of images
- Lightweight and fast (good for mobile deployment)
- Excellent feature extraction capabilities
- Saves training time and improves accuracy

**Core Functionality**: Acts as a sophisticated feature extractor that identifies patterns, edges, shapes, and objects in images without needing to train from scratch.

---

### ImageNet
**Definition**: A large database of 14+ million labeled images across thousands of categories.

**Why Used**:
- Standard dataset for training computer vision models
- Provides diverse, high-quality training data
- Models trained on ImageNet learn general visual features
- Enables transfer learning to new tasks

**Core Functionality**: Serves as the training dataset that taught MobileNetV2 to recognize visual patterns, which can then be applied to your hand sign detection task.

---

### Freeze Pre-trained Layers
**Definition**: Keeping the weights of pre-trained layers unchanged during training.

**Why Used**:
- Preserves learned features from ImageNet
- Reduces training time significantly
- Prevents destroying useful pre-trained knowledge
- Requires less training data

**Core Functionality**: Locks the feature extraction weights so only your custom classification layers learn, while leveraging existing visual understanding.

---

### Global Average Pooling 2D
**Definition**: Reduces each feature map to a single number by averaging all values.

**Why Used**:
- Dramatically reduces parameters (prevents overfitting)
- Maintains spatial information efficiently
- Replaces flattening + dense layers
- More robust to spatial variations

**Core Functionality**: Takes a 2D feature map and outputs one average value, converting spatial features into a compact representation for classification.

---

## Activation Functions

### ReLU Activation
**Definition**: Rectified Linear Unit - outputs the input if positive, zero if negative.

**Why Used**:
- Solves vanishing gradient problem
- Computationally efficient (simple max operation)
- Introduces non-linearity to learn complex patterns
- Most popular activation for hidden layers

**Core Functionality**: Acts as a simple filter that passes positive values through unchanged while blocking negative values, allowing networks to learn complex, non-linear relationships.

---

### Softmax
**Definition**: Converts raw prediction scores into probabilities that sum to 1.

**Why Used**:
- Perfect for multi-class classification
- Outputs interpretable probabilities
- Emphasizes the highest scoring class
- Works well with categorical crossentropy loss

**Core Functionality**: Transforms multiple output scores into a probability distribution where each class gets a percentage likelihood of being correct.

---

## Regularization

### Dropout
**Definition**: Randomly sets a percentage of neurons to zero during training.

**Why Used**:
- Prevents overfitting by adding randomness
- Forces network to not rely on specific neurons
- Improves generalization to new data
- Simple yet effective regularization technique

**Core Functionality**: Creates multiple "sub-networks" during training by randomly disabling connections, making the model more robust and less prone to memorization.

---

## Optimization

### Adam Optimizer
**Definition**: Adaptive learning rate optimization algorithm that combines momentum with adaptive learning rates.

**Why Used**:
- Automatically adjusts learning rate for each parameter
- Faster convergence than basic gradient descent
- Handles sparse gradients well
- Requires minimal tuning (good default choice)

**Core Functionality**: Maintains moving averages of gradients and their squares to adaptively adjust learning rates, providing faster and more stable training.

---

## Loss Functions

### Sparse Categorical Crossentropy
**Definition**: Loss function for multi-class classification with integer labels.

**Why Used**:
- Works with integer labels (0, 1, 2, 3, 4) instead of one-hot vectors
- Memory efficient for many classes
- Standard choice for classification problems
- Provides clear gradients for optimization

**Core Functionality**: Measures the difference between predicted probabilities and true class labels, providing feedback to improve model accuracy during training.

---

## Quick Reference Summary

| Term | Primary Purpose | When to Use |
|------|----------------|-------------|
| TensorFlow | Core ML framework | All deep learning projects |
| Keras | Simplified API | When you want easier coding |
| Callbacks | Training control | To automate training management |
| Layers | Data transformation | Building neural network architecture |
| MobileNetV2 | Feature extraction | Transfer learning for images |
| ImageNet | Pre-training dataset | When leveraging pre-trained models |
| Freeze layers | Preserve knowledge | Transfer learning scenarios |
| Global pooling | Dimensionality reduction | Before final classification |
| ReLU | Non-linear activation | Hidden layers in networks |
| Dropout | Prevent overfitting | When model memorizes training data |
| Softmax | Probability output | Multi-class classification |
| Adam | Optimization | Most deep learning training |
| Sparse crossentropy | Classification loss | Integer class labels |