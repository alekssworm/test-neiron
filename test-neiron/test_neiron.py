import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dims, output_dim, activations):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activations = activations
        
        self.weights = []
        self.biases = []
        
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(layer_dims) - 1):
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i+1]))
            self.biases.append(np.zeros((1, layer_dims[i+1])))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward_pass(self, X):
        activations = [X]
        
        for i in range(len(self.weights)):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if self.activations[i] == 'relu':
                activations.append(self.relu(Z))
            elif self.activations[i] == 'sigmoid':
                activations.append(self.sigmoid(Z))
        
        return activations
    
    def backward_pass(self, X, y, activations, learning_rate):
        m = X.shape[0]
        grads_weights = [np.zeros_like(w) for w in self.weights]
        grads_biases = [np.zeros_like(b) for b in self.biases]
        
        error = activations[-1] - y
        for i in range(len(grads_weights)-1, -1, -1):
            if self.activations[i] == 'relu':
                error = np.where(activations[i+1] <= 0, 0, error)
            elif self.activations[i] == 'sigmoid':
                error = error * activations[i+1] * (1 - activations[i+1])
                
            grads_weights[i] = np.dot(activations[i].T, error) / m
            grads_biases[i] = np.sum(error, axis=0, keepdims=True) / m
            
            error = np.dot(error, self.weights[i].T)
        
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_weights[i]
            self.biases[i] -= learning_rate * grads_biases[i]
    
    def train(self, X, y, epochs, batch_size, learning_rate):
        m = X.shape[0]
        for epoch in range(epochs):
            for i in range(0, m, batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                activations = self.forward_pass(batch_X)
                self.backward_pass(batch_X, batch_y, activations, learning_rate)
            
            if epoch % 100 == 0:
                loss = self.compute_loss(X, y)
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def compute_loss(self, X, y):
        activations = self.forward_pass(X)
        y_pred = activations[-1]
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def predict(self, X):
        activations = self.forward_pass(X)
        return activations[-1]

# Задаем данные для обучения
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

input_dim = 2
hidden_dims = [8, 8, 8]
output_dim = 1
activations = ['relu', 'relu', 'relu', 'sigmoid']

# Создаем и обучаем модель
model = NeuralNetwork(input_dim, hidden_dims, output_dim, activations)
model.train(X_train, y_train, epochs=1000, batch_size=4, learning_rate=0.01)

# Предсказываем результаты
predictions = model.predict(X_train)
print(predictions)
