def mse(pred, label):
    return sum([(xs - ys) ** 2 for xs, ys in zip(pred, label)]) / len(pred)


class Optimizer:
    def __init__(self, model, lr=0.01):
        self.lr = lr
        self.model = model
        self.params = model.parameters()

    def step(self):
        raise NotImplementedError("Optimizer step not implemented")

    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr})"


class SGD(Optimizer):
    def __init__(self, model, lr=0.01):
        super().__init__(model, lr=lr)

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad


class SGDMomentum(Optimizer):
    def __init__(self, model, lr=0.01, gamma=0.9):
        super().__init__(model, lr=lr)
        self.gamma = gamma
        self.velocity = {p: 0.0 for p in model.parameters()}

    def step(self):
        for param in self.params:
            self.velocity[param] = self.gamma * self.velocity[param] + param.grad
            param.data -= self.lr * self.velocity[param]
