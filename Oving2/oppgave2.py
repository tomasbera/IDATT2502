import torch
import matplotlib.pyplot as plt
import tqdm
import numpy as np

x_train = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float)
y_train = torch.tensor([[1], [1], [1], [0]], dtype=torch.float)


class NandOperator:
    def __init__(self):
        # Model variables
        self.W = torch.randn(2, 1).clone().detach().requires_grad_(True)
        self.b = torch.randn(1, 1).clone().detach().requires_grad_(True)

    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def logits(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = NandOperator()
lr = 0.1
n = 10000

optimizer = torch.optim.SGD([model.W, model.b], lr)

for epoch in tqdm.tqdm(range(n)):
    model.loss(x_train, y_train).backward()  # Computes loss gradients
    if (epoch + 1) % 100000 == 0:
        print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))
    optimizer.step()
    optimizer.zero_grad()

x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.c_[X1.ravel(), X2.ravel()]
X_grid = torch.tensor(X_grid, dtype=torch.float)


predictions = model.f(X_grid).detach().numpy().reshape(X1.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=0)  # Adjust the elevation and azimuthal angles as needed

ax.plot_surface(X1, X2, predictions, cmap='viridis')
plt.show()
