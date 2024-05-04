import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Задаем данные для обучения
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Инициализируем веса и смещения для каждого слоя
input_dim = 2
hidden1_neurons = 8
hidden2_neurons = 8
hidden3_neurons = 8
output_dim = 1

# Веса и смещения для первого скрытого слоя
W1 = np.random.randn(input_dim, hidden1_neurons)
b1 = np.zeros((1, hidden1_neurons))

# Веса и смещения для второго скрытого слоя
W2 = np.random.randn(hidden1_neurons, hidden2_neurons)
b2 = np.zeros((1, hidden2_neurons))

# Веса и смещения для третьего скрытого слоя
W3 = np.random.randn(hidden2_neurons, hidden3_neurons)
b3 = np.zeros((1, hidden3_neurons))

# Веса и смещения для выходного слоя
W4 = np.random.randn(hidden3_neurons, output_dim)
b4 = np.zeros((1, output_dim))

# Проходим по слоям нейронной сети
def forward_pass(X):
    # Первый скрытый слой
    global Z1, A1
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    # Второй скрытый слой
    global Z2, A2
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    # Третий скрытый слой
    global Z3, A3
    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)

    # Выходной слой
    global Z4, y_pred
    Z4 = np.dot(A3, W4) + b4
    y_pred = sigmoid(Z4)

    return y_pred

# Обучаем модель
def train(X, y, epochs, learning_rate):
    global W4, b4, W3, b3, W2, b2, W1, b1
    for epoch in range(epochs):
        # Прямой проход
        y_pred = forward_pass(X)

        # Вычисляем ошибку
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        # Обратное распространение ошибки
        dZ4 = y_pred - y
        dW4 = np.dot(A3.T, dZ4)
        db4 = np.sum(dZ4, axis=0, keepdims=True)

        dA3 = np.dot(dZ4, W4.T)
        dZ3 = dA3 * (Z3 > 0)
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = np.dot(dZ3, W3.T)
        dZ2 = dA2 * (Z2 > 0)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Обновляем веса
        W4 -= learning_rate * dW4
        b4 -= learning_rate * db4
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        # Выводим значение функции потерь на каждой итерации
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

# Обучаем модель
train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Предсказываем результаты
predictions = forward_pass(X_train)
print(predictions)
