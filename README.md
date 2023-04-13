# Linear Regression with PyTorch

This code demonstrates how to implement a simple linear regression model using PyTorch. The model is trained on synthetic data with a known ground truth, and the training progress is visualized with plots of the training data, test data, model predictions, and loss.

## Code Overview

### Model Definition:
>  The linear regression model is defined as a subclass of nn.Module in PyTorch. It consists of a single linear layer with one input and one output neuron.

### Data Generation:
>  Synthetic data is generated with a known ground truth. The input features X are created using the torch.arange function, and the corresponding labels y are calculated using a known weight and bias.

### Data Splitting:
>  The generated data is split into training and test sets using a 70/30 split.

> Plotting Functions: Functions for plotting the training data, test data, model predictions, and loss are defined using matplotlib.

> Model Training: The linear regression model is trained using the training data. The nn.L1Loss loss function and the optim.SGD optimizer are used for training. The training loop iterates over a specified number of epochs, computes the model's predictions, calculates the loss, performs backpropagation, and updates the model's parameters.

> Visualization of Model Training: The training progress is visualized by plotting the training data, test data, model predictions, and loss at different epochs during training.

> Model Evaluation: The trained model is evaluated on the test data, and the predictions are visualized using the plotting functions.