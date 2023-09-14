import torch
import torchvision
import matplotlib.pyplot as plt
import tqdm

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1


class WordRecognition:
    def __init__(self):
        self.W = torch.zeros([784, 10], requires_grad=True)
        self.b = torch.zeros([1, 10], requires_grad=True)

    # Predictor
    def f(self, x):
        return torch.nn.functional.softmax(self.logits(x), dim=1)

    def logits(self, x):
        return x @ self.W + self.b

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))


model = WordRecognition()
lr = 0.1
n = 1000

optimizer = torch.optim.SGD([model.W, model.b], lr)

for epoch in tqdm.tqdm(range(n)):
    model.loss(x_train, y_train).backward()  # Computes loss gradients
    if (epoch + 1) % 500 == 0:
        print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))
    optimizer.step()
    optimizer.zero_grad()

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()
y_test = torch.zeros((mnist_test.targets.shape[0], 10))
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1
print("\nAccuracy: " + str(model.accuracy(x_test, y_test)))

# Show the input of the first observation in the training set
plt.imshow(x_train[0, :].reshape(28, 28))

# Print the classification of the first observation in the training set
print(y_train[0, :])

# Save the input of the first observation in the training set
#plt.imsave('x_train_1.png', x_train[0, :].reshape(28, 28))

plt.show()

fig = plt.figure('Photos')
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(model.W[:, i].detach().numpy().reshape(28, 28))
    plt.title(f'W: {i}')
    plt.xticks([])
    plt.yticks([])

plt.show()


