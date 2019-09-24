# ml-fun
Fun machine learning stuff

## Feedforward neural network

`model.py` contains a dynamic neural network made with NumPy, which can be used for classification (required in one of the machine learning exercise at the University). This is not an attempt to reinvent the wheel, since neural nets are efficiently implemented by frameworks like TF, but rather to review the mathematical background behind it.

`example.py` builds a neural network of the class `NN`, described in `model.py`, and trains it on the Iris dataset to distinguish the type "versicolor" from the other types based on the attributes petal length and petal width.
