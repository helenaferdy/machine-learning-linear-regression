import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# torch.manual_seed(42)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

weight = 0.7
bias = 0.3
X = torch.arange(start=0, end=2, step=0.02).unsqueeze(dim=1)
y = weight * X + bias
train_split = int(0.7 * len(X))
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

def plot_train(train_data=X_train, train_label=y_train, test_data=X_test, test_label=y_test, train_model=None, test_model=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_label, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_label, c="g", s=4, label="Test Data")
    if train_model is not None:
        plt.scatter(train_data, train_model, c="r", s=4, label="Model Training")
    if test_model is not None:
        plt.scatter(test_data, test_model, c="c", s=4, label="Model Testing")
    plt.legend(prop={"size":14})
    plt.show()

def plot_loss():
    plt.plot(epoch_count, losses, linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Plot Title')
    plt.gca().invert_yaxis()
    plt.show()


model = LinearRegressionModel()
loss_fn = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []
epoch_count = []
epochs = 1000
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      losses.append(f'{loss.item():.4f}')    
      epoch_count.append(epoch +1) 
      print(f'Epoch : [{epoch+1}/{epochs}], Loss : {loss.item():.4f}')

with torch.inference_mode():
    for name, param in model.named_parameters():
        print(f'{name}: {param.item():.4f}')

with torch.inference_mode():
    y_pred = model(X_train)
    y_pred_test = model(X_test)
    plot_train(train_model=y_pred, test_model=y_pred_test)
    print("")
    plot_loss()