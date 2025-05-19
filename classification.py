import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons

from src.nn import MLP
from src.optimizer import SGD, mse

X, y = make_moons(n_samples=200, shuffle=True, noise=0.15, random_state=42)
y = y * 2 - 1


def model_predict_visualize_custom(X, y, net=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")

    if net is not None:
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx1, xx2 = np.meshgrid(
            np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)
        )
        X_grid = np.stack([xx1.ravel(), xx2.ravel()], axis=1)
        # Predict using the custom MLP
        y_grid = []
        for xg in X_grid:
            out = net(xg.tolist())
            # Output is a Value object, get its data and sign
            y_grid.append(np.sign(out.data))
        y_grid = np.array(y_grid).reshape(xx1.shape)
        plt.contourf(xx1, xx2, y_grid, cmap="jet", alpha=0.2)

    plt.show()


# Converting to list as we don't support numpy
xs, ys = X.tolist(), y.tolist()


mlp = MLP(2, [4, 1])
lr = 0.01
epochs = 500
losses = []
optimizer = SGD(mlp, lr=lr)
print(optimizer)


for epoch in range(epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    pred = []
    for x in xs:
        pred.append(mlp(x))

    loss = mse(pred, ys)
    p = [1 if p.data > 0 else -1 for p in pred]
    acc = np.mean(np.array(p) == np.array(ys))
    loss.backward()

    # Update
    optimizer.step()

    losses.append(loss.data)
    print(f"Epoch: {epoch + 1}, Loss: {loss}, Accuracy: {acc}")

# Visualize the results
model_predict_visualize_custom(X, y, mlp)
