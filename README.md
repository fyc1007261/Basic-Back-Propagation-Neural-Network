# Back-Propagation Neural Network

By Cai Yifan (student id = 516030910375)

### 1. Functions

- The project provides a basic package of back-propagation neural network.
- The following criteria are achieved.
  - [x] A variable number of hidden layers;
  - [x] A variable number of nodes in each hidden layer;
  - [x] Variable activation functions (sigmoid, relu and tanh);
  - [x] Matrix and vector representation of weights and data;
  - [x] Variable dimensions of input layer and output layer;
  - [ ] Data from UCI ML Repository
        - Unfortunately, due to limited time, the network only supports MSE (mean square error) as the loss function;
        - However, using MSE is not proper in classification problems;
        - Instead, random data with a specific relationship is generated and used in the example code.

### 2. Usage

#### 2.1. Dependencies

- `numpy` and `pandas` are used in this project.
- `numpy` is used for matrix  representation and computing; pandas is used for load data for files.
- Python 3.6 is used as the interpreter.

#### 2.2. Interfaces of the package

- `network.__init__(inp)`: 
  - Initialize a network with `inp` as the input data.
  - No value returns.
- `network.add_layer(units, activation=None)`: 
  - Add a hidden or output layer. The number of nodes is specified by `units` and the activation function by `activation`. 
  - The `activation` supports `None`, `'sigmoid'`, `'tanh'` or `'relu'`. 
  - No value returns.
-  `network.build(labels)`: 
  - The network should be built after all hidden layers have been configured and before forwarding or backwarding.
  - The outputs of input data `inp` are specified by `labels`.
  - No value returns.
- `network.forward()`: 
  - Do forwarding operations of the network. 
  - This method returns the current predictions of the network.
- `network.backward(learning_rate=0.01)`:
  - Do backward operations of the network.
  - **Forward method must be completed before each backward by the user!**
  - Returns the MSE (mean square error) **before** the backwarding.

#### 2.3. Example code

```python
# The code is provided in model.py
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
        print("Epoch=",i,'/',epochs, 'Loss=', loss)
        
    # print the final predictions
    pred = NN.forward()
    print(pred)

```

