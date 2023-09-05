import torch
import matplotlib.pyplot as plt
import csv

days = []
length = []
weight = []

with open('day_length_weight.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)

    for row in csvreader:
        days.append(float(row['day']))
        length.append(float(row['length']))
        weight.append(float(row['weight']))

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

days_train = torch.tensor(days).reshape(-1, 1)
combined_arr = list(zip(length, weight))
weight_length_train = torch.tensor(combined_arr).reshape(-1, 2)

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(50000):
    model.loss(weight_length_train, days_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(weight_length_train, days_train)))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(length, weight, days_train, label='$(x^{(i)},y^{(i)},z^{(i)})$')
ax.scatter(length, weight, model.f(weight_length_train).detach(), label='$\\hat y = f(x) = xW+b$', color='red')
ax.legend()
plt.show()
