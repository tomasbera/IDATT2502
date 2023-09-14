import torch
import matplotlib.pyplot as plt
import tqdm
import numpy as np

x_train = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float).reshape(-1,2)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

converge = False

if(converge):
    W1_init = np.array([[10.0, -10.0], [10.0, -10.0]])
    b1_init = np.array([[-5.0, 15.0]])
    W2_init = np.array([[10.0], [10.0]])
    b2_init = np.array([[-15.0]])
else:
    W1_init = torch.randn(2, 2).clone().detach().requires_grad_(True)
    b1_init = torch.randn(1, 2).clone().detach().requires_grad_(True)
    W2_init = torch.randn(2, 1).clone().detach().requires_grad_(True)
    b2_init = torch.randn(1, 1).clone().detach().requires_grad_(True)


class XOROperator:
    def __init__(self, W1=W1_init, W2=W2_init, b1=b1_init, b2=b2_init):
        self.W1 = torch.tensor(W1, dtype=torch.float, requires_grad=True)
        self.W2 = torch.tensor(W2, dtype=torch.float, requires_grad=True)
        self.b1 = torch.tensor(b1, dtype=torch.float, requires_grad=True)
        self.b2 = torch.tensor(b2, dtype=torch.float, requires_grad=True)

    # First layer function
    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    # Second layer function
    def f2(self, h):
        return torch.sigmoid(h @ self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


model = XOROperator()
lr = 0.1
n = 100000

optimizer = torch.optim.SGD([model.W2, model.b2, model.W1, model.b1], lr)

for epoch in tqdm.tqdm(range(n)):
    model.loss(x_train, y_train).backward()  # Computes loss gradients
    if epoch % 100000 == 0:
        print("W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s"
              % (model.W1.data, model.b1.data, model.W2.data, model.b2.data, model.loss(x_train, y_train).data))
    optimizer.step()
    optimizer.zero_grad()

print("W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s"
      % (model.W1.data, model.b1.data, model.W2.data, model.b2.data, model.loss(x_train, y_train).data))


x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.c_[X1.ravel(), X2.ravel()]
X_grid = torch.tensor(X_grid, dtype=torch.float)

predictions = model.f(X_grid).detach().numpy().reshape(X1.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, predictions, cmap='viridis')
plt.show()
