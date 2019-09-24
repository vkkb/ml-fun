from model import NN
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

# Set seed to make results reproducible
seed = 100
np.random.seed(seed)

# Load data
iris_data = datasets.load_iris()
target = iris_data.target

# Change target to: 1 if versicolor, else 0
target = [1 if 1 == x else 0 for x in target]
t = np.array(target)[:, np.newaxis]

# Use only the attributes "Petal Length and Petal Width"
X = iris_data.data[:, 2:]
plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=t.flatten())
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Real Data', fontsize=16)
plt.show()

# Build neural network
model = NN()
model.add_layer(units=10, activation=np.tanh, initialization=np.random.normal)
model.add_layer(units=5, activation=np.tanh, initialization=np.random.normal)
model.add_layer(units=2, activation=np.tanh, initialization=np.random.normal)
model.add_layer(1, activation=model.sigmoid)

model.train(X, t, iterations=5000, learn_rate=0.0001)  # 5000
prediction = model.predict(X)

# Plot error
plt.plot(model.history['epoch'], model.history['error_train'],
         label='Training error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.show()

# Plot prediction
plt.scatter(X[:, 0], X[:, 1], c=np.round(prediction, 0).flatten())
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Predictions', fontsize=16)
plt.show()
