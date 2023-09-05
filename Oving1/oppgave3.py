import torch
import matplotlib.pyplot as plt
import csv

days = []
head_circumference = []

# Open the CSV file
with open('day_head_circumference.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)

    # Iterate over each row in the CSV file
    for row in csvreader:
        days.append(float(row['day']))
        head_circumference.append(float(row['head_circumference']))

# Observed/training input and output
x_train = torch.tensor(days).reshape(-1, 1)
y_train = torch.tensor(head_circumference).reshape(-1, 1)


class NoneLinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20*torch.sigmoid(x @ self.W + self.b) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = NoneLinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.00001)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(torch.min(x_train), torch.max(x_train), 1.0).reshape(-1, 1)
y = model.f(x).detach()
plt.plot(x, y, label='$f(x) = 20\sigma(xW+b) + 31$')
plt.legend()
plt.show()
