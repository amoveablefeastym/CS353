import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

def analyzeResults(X_test, y_test, model):
    '''
    This function analyzes the results of the trained model.
    input:
    X_test: input array, has shape [numInputFeatures, numExamples] = [2, 100]
    y_test: true labels, has shape [numOutputUnits, numExamples] = [1, 100]
    model: trained model.
    '''
    y_predicted = model.predict(X_test)
    y_predicted = y_predicted.flatten()
    y_test = y_test.flatten()
    accuracy = np.mean(y_predicted == y_test)
    print('Accuracy:', accuracy)
    plotDecisionBoundary(X_test, y_test, model)
    
def generateSpirals(N:int):
    # ALREADY IMPLEMENTED
    # Found this at: https://gist.github.com/45deg/e731d9e7f478de134def5668324c44c5
    theta = np.sqrt(np.random.rand(N))*2*np.pi # np.linspace(0,2*pi,100)

    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + 1.2*np.random.randn(N,2)

    r_b = -2*theta - np.pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + 1.2*np.random.randn(N,2)

    res_a = np.append(x_a, np.zeros((N,1)), axis=1)
    res_b = np.append(x_b, np.ones((N,1)), axis=1)
    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)
    return res[:, :2], res[:, 2].reshape(1, -1)

def loadDataset():
    '''
    This loads the dataset for you to train your neural network on.
    output:
    X: input array, has shape [numInputFeatures, numExamples] = [2, 300]
    y: true labels, has shape [numOutputUnits, numExamples] = [1, 300]
    X_test: input array, has shape [numInputFeatures, numExamples] = [2, 100]
    y_test: true labels, has shape [numOutputUnits, numExamples] = [1, 100]
    '''
    # ALREADY IMPLEMENTED
    np.random.seed(42)
    #X, y = make_moons(n_samples=300, noise=0.15)
    X, y = generateSpirals(300)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X-mean)/std
    #X_test, y_test = make_moons(n_samples=100, noise=0.15)
    X_test, y_test = generateSpirals(100)
    X_test = (X_test-mean)/std
    X, X_test = X.T, X_test.T
    y, y_test = y.reshape(1, -1), y_test.reshape(1, -1)
    return X, y, X_test, y_test

def displayData(X, y, title:str='Data points'):
    '''
    This function displays the data points.
    input:
    X: input array, has shape [numInputFeatures, numExamples] = [2, 300]
    y: true labels, has shape [numOutputUnits, numExamples] = [1, 300]
    '''
    # ALREADY IMPLEMENTED
    plt.scatter(X[0, :], X[1, :], c=y.flatten(), cmap=plt.cm.Spectral)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

def plotDecisionBoundary(X, y, model):
    # ALREADY IMPLEMENTED
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.title('Decision Boundary')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()

def sigmoid(z:np.ndarray) -> np.ndarray:
    '''
    Part of Question 1 Part a
    This function returns the element-wise sigmoid of the input array.
    input:
    z: input array, weighted sums.
    output:
    a: output array, activations. 
    '''
    # YOUR CODE HERE
    a = 1/(1 + np.exp(-z))
    # YOUR CODE ENDS HERE
    return a

def sigmoidDerivative(z:np.ndarray) -> np.ndarray:
    '''
    Part of Question 1 Part b
    This function returns the element-wise derivative of the sigmoid function.
    input:
    z: input array, activations.
    output:
    dx: output array, derivative of the sigmoid function.
    '''
    # YOUR CODE HERE
    da = sigmoid(z)*(1 - sigmoid(z))
    # YOUR CODE ENDS HERE
    return da

def L2Loss(y:np.ndarray, y_hat:np.ndarray) -> float:
    '''
    Part of Question 1 Part b
    This function returns the L2 loss between the predicted and true labels.
    input:
    y: true labels, has shape [numOutputUnits, numExamples].
    y_hat: predicted labels, has shape [numOutputUnits, numExamples].
    output:
    loss: L2 loss between y and y_hat. Normalized by the number of examples.
    '''
    # YOUR CODE HERE
    loss = None
    # YOUR CODE ENDS HERE
    return loss

def plotTrainingLoss(lossList:np.ndarray):
    '''
    This function plots the training loss.
    input:
    lossList: list of L2 loss values at each training step.
    '''
    # ALREADY IMPLEMENTED
    plt.plot(lossList)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


class FCNN:
    def __init__(self, input_size:int, hidden_size:int, output_size:int, init:str='random', AutograderSeed:int=0):
        '''
        Part of Question 1 Part a
        This function initializes the weights and biases of the neural network.
        input:
        input_size: number of input features.
        hidden_size: number of hidden units.
        output_size: number of output units.
        init: initialization method for the weights:
            'random': random normal distribution with mean 0 and standard deviation 1.
            'zeros': all zeros.
        '''
        np.random.seed(AutograderSeed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # YOUR CODE HERE (PART OF QUESTION 1 PART A)
        if init == 'random':
            # Initilize weights using random normal distribution with mean 0 and standard deviation 1.
            self.W1 = np.random.normal(0,1,(hidden_size, input_size))
            self.W2 = np.random.normal(0,1,(output_size, hidden_size))
        elif init == 'zeros':
            # Initialize weights to zeros.
            self.W1 = np.zeros((hidden_size, input_size))
            self.W2 = np.zeros((output_size, hidden_size))
        self.numTrainableParams = None
        # YOUR CODE ENDS HERE (PART OF QUESTION 1 PART A)
    
    def forward(self, x:np.ndarray, train:bool=False) -> np.ndarray:
        '''
        Part of Question 1 Part a
        This function performs the forward pass of the neural network.
        input:
        x: input array, has shape [numInputFeatures, numExamples].
        train: boolean flag, if True, the forward pass is being performed during training.
        output:
        output: output array, has shape [numOutputUnits, numExamples].
        forwardPassResults: dictionary containing intermediate results of the forward pass.
        Compute this only during training. You need to return the following:
            forwardPassResults['x']: input array.
            forwardPassResults['z1']: weighted sums of the first layer.
            forwardPassResults['a1']: activations of the first layer.
            forwardPassResults['z2']: weighted sums of the second layer.
            forwardPassResults['y']: output array.
        We will use these intermediate results in the backward pass.
        '''
        # YOUR CODE HERE (PART OF QUESTION 1 PART A)
        forwardPassResults = {}
        if train:
            forwardPassResults['x'] = x
            forwardPassResults['z1'] = x*self.W1.T
            forwardPassResults['a1'] = sigmoid(forwardPassResults['z1'])
            forwardPassResults['z2'] = forwardPassResults['a1']*self.W2.T
            forwardPassResults['y'] = sigmoid(forwardPassResults['z2'])
            y = forwardPassResults['y']
        else:
            y = None

        # YOUR CODE ENDS HERE (PART OF QUESTION 1 PART A)
        if train:
            return y, forwardPassResults
        else:
            return y

    def backward(self, forwardPassResults, y_predicted:np.ndarray, y_true:np.ndarray) -> dict:
        '''
        Part of Question 1 Part b
        This function performs the backward pass of the neural network. Computes gradients with respect to the weights.
        input:
        forwardPassResults: dictionary containing intermediate results of the forward pass.
        y_predicted: predicted labels, has shape [numOutputUnits, numExamples].
        y_true: true labels, has shape [numOutputUnits, numExamples].
        output:
        gradients: dictionary containing gradients of the weights. For this question it should have two parts:
            gradients['dW1']: gradient of the weights of the first layer.
            gradients['dW2']: gradient of the weights of the second layer.
        '''
        gradients = {}
        # YOUR CODE HERE (PART OF QUESTION 1 PART B)

        gradients['dW1'] = None
        gradients['dW2'] = None
        # YOUR CODE ENDS HERE (PART OF QUESTION 1 PART B)
        return gradients

    def updateWeights(self, gradients:dict, learning_rate:float):
        '''
        Part of Question 1 Part C
        This function updates the weights of the neural network using the gradients.
        input:
        gradients: dictionary containing gradients of the weights.
        learning_rate: learning rate for the update.
        '''
        # YOUR CODE HERE (PART OF QUESTION 1 PART B)

        # YOUR CODE ENDS HERE (PART OF QUESTION 1 PART B)

    def predict(self, x:np.ndarray) -> np.ndarray:
        '''
        Part of Question 1 Part C
        This function predicts the labels of the input data.
        If network output is bigger than 0.5, predict 1, else predict 0.
        input:
        x: input array, has shape [numInputFeatures, numExamples].
        output:
        y: predicted labels, has shape [numOutputUnits, numExamples].
        '''
        # YOUR CODE HERE (PART OF QUESTION 1 PART C)
        y = None
        # YOUR CODE ENDS HERE (PART OF QUESTION 1 PART C)
        return y
    
    def train(self, examples: np.ndarray, labels: np.ndarray, numSteps: int, learning_rate: float):
        '''
        In this function we train the neural network using the examples and labels.
        input:
        examples: input array, has shape [numInputFeatures, numExamples]. ([2 by numExamples])
        labels: true labels, has shape [numOutputUnits, numExamples]. ([1 by numExamples])
        max_steps: number of gradient training steps.
        learning_rate: learning rate for the update equation.
        output:
        lossList: list of L2 loss values at each training step.
        '''
        lossList = []
        loss = L2Loss(self.forward(examples), labels)
        for i in range(numSteps):
            y_predicted, forwardPassResults = self.forward(examples, train=True)
            gradients = self.backward(forwardPassResults, y_predicted, labels)
            self.updateWeights(gradients, learning_rate)
            loss = L2Loss(y_predicted, labels)
            lossList.append(loss)
        return lossList
