import numpy as np

def leaky_relu(x):
    return np.maximum(0.01*x,x)

def leaky_relu_dervative(x):
    return np.where(x <= 0,0.01,1)

inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 5, 32, 1
hidden_weights = np.random.uniform(low=-1,high=1,size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias = np.random.uniform(low=-1,high=1,size=(1,hiddenLayerNeurons))
output_weights = np.random.uniform(low=-1,high=1,size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(low=-1,high=1,size=(1,outputLayerNeurons))
epochs = 100
lr = 0.1

def forward_pass(X):
    z_hidden_layer = X.dot(hidden_weights) + hidden_bias
    y_hidden_layer = leaky_relu(z_hidden_layer)

    z_output_layer = y_hidden_layer.dot(output_weights) + output_bias
    y_output_layer = leaky_relu(z_output_layer)

    return z_hidden_layer, y_hidden_layer, z_output_layer, y_output_layer

titanic_X = np.load('data/titanic_X_train.npy')
titanic_y = np.load('data/titanic_y_train.npy')

for epoch in range(epochs):
    print(epoch)
    error = 0
    for (X,y) in zip(titanic_X,titanic_y):
        X = X.reshape(1,5)
        z_hidden_layer, y_hidden_layer, z_output_layer, y_output_layer = forward_pass(X)
        error = y - y_output_layer
        d_predicted_output = error * leaky_relu_dervative(y_output_layer)

        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * leaky_relu_dervative(y_hidden_layer)

        output_weights += y_hidden_layer.T.dot(d_predicted_output) * lr
        output_bias += d_predicted_output * lr
        hidden_weights += X.T.dot(d_hidden_layer) * lr
        hidden_bias += d_hidden_layer.mean() * lr
    print(d_hidden_layer)

titanic_X_test = np.load('data/titanic_X_test.npy')
titanic_y_test = np.load('data/titanic_y_test.npy')

counter_total = 0
counter_correct = 0

for (X, y) in zip(titanic_X_test, titanic_y_test):
    counter_total += 1
    z_hidden_layer, y_hidden_layer, z_output_layer, y_output_layer = forward_pass(X)
    if((y_output_layer.flatten()[0] >= 0.5) and y == 1):
        counter_correct += 1
    if ((y_output_layer.flatten()[0] < 0.5) and y == 0):
        counter_correct += 1
print(counter_correct)
print((counter_correct / counter_total)*100)






