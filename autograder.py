import numpy as np
import pickle
import q1
import q2


def test_sigmoid():
    np.random.seed(0)
    with open('autograderExpectedOutputs.pkl', 'rb') as f:
        expectedOutput = pickle.load(f)
    expectedOutput = expectedOutput['sigmoid']
    x = np.random.randn(10)
    output = q1.sigmoid(x)
    if np.allclose(output, expectedOutput):
        print('Sigmoid: PASS')
    else:
        print('Sigmoid: FAIL')

def test_networkInitialization():
    with open('autograderExpectedOutputs.pkl', 'rb') as f:
        expectedOutput = pickle.load(f)
    expectedOutput = expectedOutput['networkInitialization']
    ##### TEST CASE 1 #####
    network = q1.FCNN(2, 5, 1)
    if not np.allclose(expectedOutput[0]['W1'], network.W1):
        print('Network Random Initialization Failed (W1)')
        return
    if not np.allclose(expectedOutput[0]['W2'], network.W2):
        print('Network Random Initialization Failed (W2)')
        return
    if not np.allclose(expectedOutput[0]['numTrainableParams'], network.numTrainableParams):
        print('Network Random Initialization Failed (numTrainableParams)')
        return
    #### TEST CASE 2 ####
    network = q1.FCNN(2, 8, 1, init='zeros')
    if not np.allclose(expectedOutput[1]['W1'], network.W1):
        print('Network Zero Initialization Failed (W1)')
        return
    if not np.allclose(expectedOutput[1]['W2'], network.W2):
        print('Network Zero Initialization Failed (W2)')
        return
    if not np.allclose(expectedOutput[1]['numTrainableParams'], network.numTrainableParams):
        print('Network Zero Initialization Failed (numTrainableParams)')
        return
    print('Network Initialization: PASS')

def test_forward():
    np.random.seed(0)
    with open('autograderExpectedOutputs.pkl', 'rb') as f:
        expectedOutput = pickle.load(f)
    expectedOutput = expectedOutput['forward']
    network = q1.FCNN(2, 5, 1)
    x = np.random.randn(2, 10)
    output, dict = network.forward(x, train=True)
    if not np.allclose(output, expectedOutput['output']):
        print('Forward: FAIL (OUTPUT)')
        return
    if not np.allclose(dict['a1'], expectedOutput['a1']):
        print('Forward: FAIL (a1)')
        return
    if not np.allclose(dict['z1'], expectedOutput['z1']):
        print('Forward: FAIL (z1)')
        return
    if not np.allclose(dict['z2'], expectedOutput['z2']):
        print('Forward: FAIL (z2)')
        return
    if not np.allclose(dict['x'], expectedOutput['x']):
        print('Forward: FAIL (x)')
        return
    if not np.allclose(dict['y'], expectedOutput['y']):
        print('Forward: FAIL (y)')
        return
    print('Forward: PASS')
    

def test_L2Loss():
    np.random.seed(0)
    a = np.random.randn(1,10)
    b = np.random.randn(1,10)
    with open('autograderExpectedOutputs.pkl', 'rb') as f:
        expectedOutput = pickle.load(f)
    expectedOutput = expectedOutput['L2Loss']
    output = q1.L2Loss(a, b)
    if np.allclose(output, expectedOutput):
        print('L2 Loss: PASS')
    else:
        print('L2 Loss: FAIL')

def test_sigmodDerivative():
    np.random.seed(0)
    with open('autograderExpectedOutputs.pkl', 'rb') as f:
        expectedOutput = pickle.load(f)
    expectedOutput = expectedOutput['sigmoidDerivative']
    x = np.random.randn(10)
    output = q1.sigmoidDerivative(x)
    if np.allclose(output, expectedOutput):
        print('Sigmoid Derivative: PASS')
    else:
        print('Sigmoid Derivative: FAIL')

def test_backward():
    np.random.seed(0)
    with open('autograderExpectedOutputs.pkl', 'rb') as f:
        expectedOutput = pickle.load(f)
    expectedOutput = expectedOutput['backward']
    network = q1.FCNN(2, 5, 1, AutograderSeed=0)
    x = np.random.randn(2, 10)
    y = np.random.randn(1, 10)
    output, dict = network.forward(x, train=True)
    grads = network.backward(dict, output, y)
    if not np.allclose(grads['dW1'], expectedOutput['dW1']):
        print('Backward: FAIL (dW1)')
        return
    if not np.allclose(grads['dW2'], expectedOutput['dW2']):
        print('Backward: FAIL (dW2)')
        return
    print('Backward: PASS')

def test_updateWeights():
    np.random.seed(0)
    with open('autograderExpectedOutputs.pkl', 'rb') as f:
        expectedOutput = pickle.load(f)
    expectedOutput = expectedOutput['updateWeights']
    network = q1.FCNN(2, 5, 1)
    grads = {
        'dW1': np.random.randn(5, 2),
        'dW2': np.random.randn(1, 5)
    }
    network.updateWeights(grads, 0.1)
    if not np.allclose(network.W1, expectedOutput['W1']):
        print('Update Weights: FAIL (W1)')
        return
    if not np.allclose(network.W2, expectedOutput['W2']):
        print('Update Weights: FAIL (W2)')
        return
    print('Update Weights: PASS')

def test_predict():
    np.random.seed(0)
    with open('autograderExpectedOutputs.pkl', 'rb') as f:
        expectedOutput = pickle.load(f)
    expectedOutput = expectedOutput['predict']
    network = q1.FCNN(2, 5, 1)
    x = np.random.randn(2, 10)
    output = network.predict(x)
    if not np.allclose(output, expectedOutput):
        print('Predict: FAIL')
        return
    print('Predict: PASS')

def test_assignClusters():
    np.random.seed(0)
    with open('autograderExpectedOutputs.pkl', 'rb') as f:
        expectedOutput = pickle.load(f)
    expectedOutput = expectedOutput['assignClusters']
    data = np.load('kmeansdata.npz')
    data = np.stack((data['x'], data['y']), axis=0)
    k=3
    clusters = data[:, np.random.choice(data.shape[1], k, replace=False)]
    output = q2.assignClusters(data, clusters)
    if np.allclose(output, expectedOutput):
        print('Assign Clusters: PASS')
    else:
        print('Assign Clusters: FAIL')
    if output.dtype != np.int32 and output.dtype != np.int64:
        print('''Warning: You pass the test but the output is not of type int32 or int64. Please make sure the output 
              is of type int32 or int64. This might be an issue for the visualization function provided for you. ''')
        print(output.dtype)
    return

def test_updateClusters():
    np.random.seed(0)
    with open('autograderExpectedOutputs.pkl', 'rb') as f:
        expectedOutput = pickle.load(f)
    expectedOutput = expectedOutput['updateClusters']
    data = np.load('kmeansdata.npz')
    data = np.stack((data['x'], data['y']), axis=0)
    assignments = np.random.randint(0, 3, 150)
    k = 3
    output = q2.updateClusters(data, assignments, k)
    if np.allclose(output, expectedOutput):
        print('Update Clusters: PASS')
    else:
        print('Update Clusters: FAIL')

def gradeAll():
    test_sigmoid()
    test_networkInitialization()
    test_forward()
    test_L2Loss()
    test_sigmodDerivative()
    test_backward()
    test_updateWeights()
    test_predict()
    test_assignClusters()
    test_updateClusters()

def generateExpectedOutput():
    expectedOutput = {}
    ##### Sigmoid #####
    np.random.seed(0)
    x = np.random.randn(10)
    expectedOutput['sigmoid'] = q1.sigmoid(x)
    ##### Network Initialization #####
    network = q1.FCNN(2, 5, 1)
    expectedOutput['networkInitialization'] = []
    expectedOutput['networkInitialization'].append({
        'W1': network.W1,
        'W2': network.W2,
        'numTrainableParams': network.numTrainableParams
    })
    network = q1.FCNN(2, 8, 1, init='zeros')
    expectedOutput['networkInitialization'].append({
        'W1': network.W1,
        'W2': network.W2,
        'numTrainableParams': network.numTrainableParams
    })
    ##### Forward #####
    np.random.seed(0)
    network = q1.FCNN(2, 5, 1)
    x = np.random.randn(2, 10)
    output, dict = network.forward(x, train=True)
    expectedOutput['forward'] = {
        'output': output,
        'a1': dict['a1'],
        'z1': dict['z1'],
        'z2': dict['z2'],
        'x': dict['x'],
        'y': dict['y']
    }
    ##### L2 Loss #####
    np.random.seed(0)
    a = np.random.randn(1, 10)
    b = np.random.randn(1, 10)
    expectedOutput['L2Loss'] = q1.L2Loss(a, b)
    print(q1.L2Loss(a, b))
    ##### Sigmoid Derivative #####
    np.random.seed(0)
    x = np.random.randn(10)
    expectedOutput['sigmoidDerivative'] = q1.sigmoidDerivative(x)
    ##### Backward #####
    np.random.seed(0)
    network = q1.FCNN(2, 5, 1, AutograderSeed=0)
    x = np.random.randn(2, 10)
    y = np.random.randn(1, 10)
    output, dict = network.forward(x, train=True)
    grads = network.backward(dict, output, y)
    expectedOutput['backward'] = {
        'dW1': grads['dW1'],
        'dW2': grads['dW2']
    }
    ##### Update Weights #####
    np.random.seed(0)
    network = q1.FCNN(2, 5, 1)
    grads = {
        'dW1': np.random.randn(5, 2),
        'dW2': np.random.randn(1, 5)
    }
    network.updateWeights(grads, 0.1)
    expectedOutput['updateWeights'] = {
        'W1': network.W1,
        'W2': network.W2
    }
    ##### Predict #####
    np.random.seed(0)
    network = q1.FCNN(2, 5, 1)
    x = np.random.randn(2, 10)
    expectedOutput['predict'] = network.predict(x)
    ##### Assign Clusters #####
    np.random.seed(0)
    data = np.load('kmeansdata.npz')
    data = np.stack((data['x'], data['y']), axis=0)
    k=3
    clusters = data[:, np.random.choice(data.shape[1], k, replace=False)]
    expectedOutput['assignClusters'] = q2.assignClusters(data, clusters)
    ##### Update Clusters #####
    np.random.seed(0)
    data = np.load('kmeansdata.npz')
    data = np.stack((data['x'], data['y']), axis=0)
    assignments = np.random.randint(0, 3, 150)
    k = 3
    expectedOutput['updateClusters'] = q2.updateClusters(data, assignments, k)
    with open('autograderExpectedOutputs.pkl', 'wb') as f:
        pickle.dump(expectedOutput, f)
    return
    

