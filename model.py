from julia import Main
Main.include("ops.jl")

class Activations:
    @staticmethod
    def linear(x):
        return Main.Activations.linear(x)

    @staticmethod
    def relu(x):
        return Main.Activations.relu(x)

    @staticmethod
    def sigmoid(x):
        return Main.Activations.sigmoid(x)

    @staticmethod
    def tanh(x):
        return Main.Activations.tanh(x)



class DenseLayer:
    def __init__(self, w, b, optimizer = None):
        self.w = w
        self.b = b
        self.optimizer = optimizer

    def forward(self, x):
        self.x = x
        return Main.Layers.forward(x, self.w, self.b)
    
    def backward(self, dz):
        dx, dw, db = Main.Layers.backward(dz, self.x, self.w)
        if self.optimizer is not None:
            self.optimizer.update_params(self.w, self.b, dw, db)
        return dx
    
