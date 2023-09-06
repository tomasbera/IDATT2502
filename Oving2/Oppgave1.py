import torch
import matplotlib.pyplot as plt
import tqdm

x_train = torch.tensor([[0.0], [1.0]], dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor([[1], [0]], dtype=torch.float).reshape(-1, 1)


class NotOperator:

    def __init__(self):
        # Model variables
        self.W = torch.randn(1, 1).clone().detach().requires_grad_(True)
        self.b = torch.randn(1, 1).clone().detach().requires_grad_(True)

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def logits(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)



model = NotOperator()

optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in tqdm.tqdm(range(100000)):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
x = torch.arange(0.0, 1.0, 0.01).reshape(-1, 1)
plt.plot(x, model.f(x).detach(), label='$y=f(x)=1/(1+np.exp(-z))')
plt.legend()
plt.show()
