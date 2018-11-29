import network
import numpy as np
import pandas as pd

# example code for the network

if __name__ == "__main__":

    # load the data from local files
    data = np.array(pd.read_csv('data/data.csv').T)
    label = np.array(pd.read_csv('data/label.csv').T)

    # create a class of the neural network
    NN = network.BackPropNetwork(data)

    # Add a layer with 4 nodes and sigmoid as the activation function,
    # then add a layer with 3 nodes and tanh as the activation.
    NN.add_layer(4, 'sigmoid')
    NN.add_layer(3, 'tanh')

    # Add the output layer. The dimension of the output is 1.
    # As we are doing regression, no activation should be in the output layer.
    NN.add_layer(1)

    # build the model
    NN.build(label)

    # train the model for 400 epochs
    epochs = 400
    for i in range(epochs):
        # forwarding is required before backwarding
        NN.forward()
        # set the learning rate to be 0.01
        loss = NN.backward(learning_rate=0.01)
        print("Epoch=", i, '/', epochs, 'Loss=', loss)

    # print the final predictions
    pred = NN.forward()
    print(pred)
