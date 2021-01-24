import numpy as np
import matplotlib.pyplot as plt
import argparse

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    e_x = np.exp(x - (np.max(x, axis=1)).reshape(x.shape[0], 1))
    p = e_x / np.sum(e_x, axis=1, keepdims=True)
    assert (p.shape[0] == x.shape[0]) and (p.shape[1] == x.shape[1])
    return p
    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    def _positive_sigmoid(_x):
        return 1 / (1 + np.exp(-_x))

    def _negative_sigmoid(_x):
        exp = np.exp(_x)
        return exp / (exp + 1)

    positive = x >= 0
    negative = ~positive
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(_x=x[positive])
    result[negative] = _negative_sigmoid(_x=x[negative])
    return result
    # *** END CODE HERE ***

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """
    # *** START CODE HERE ***
    params = dict()
    params['W1'] = np.random.normal(loc=0.0, scale=1.0, size=(input_size, num_hidden))
    params['b1'] = np.zeros(num_hidden)
    params['W2'] = np.random.normal(loc=0.0, scale=1.0, size=(num_hidden, num_output))
    params['b2'] = np.zeros(num_output)
    return params
    # *** END CODE HERE ***

def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    a_1 = np.dot(data, params['W1']) + params['b1']
    z_1 = sigmoid(a_1)
    a_2 = np.dot(z_1, params['W2']) + params['b2']
    output = softmax(a_2)

    avg_loss = 0.0
    sample_size = labels.shape[0]
    number_of_class = labels.shape[1]

    for i in range(sample_size):
        for j in range(number_of_class):
            avg_loss = avg_loss + labels[i, j]*np.log(output[i, j])
    avg_loss = -1.0 * avg_loss/sample_size
    return z_1, output, avg_loss
    # *** END CODE HERE ***

def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    z_1, output, _ = forward_prop_func(data, labels, params)
    sample_size = labels.shape[0]
    gradients = dict()

    dscores = (output - labels)/sample_size

    gradients['W2'] = np.dot(z_1.T, dscores)
    gradients['b2'] = np.sum(dscores, axis=0, keepdims=True)

    dhidden = np.dot(dscores, params['W2'].T)
    dhidden = np.multiply(dhidden, z_1*(1.0 - z_1))
    gradients['W1'] = np.dot(data.T, dhidden)
    gradients['b1'] = np.sum(dhidden, axis=0, keepdims=True)
    return gradients
    # *** END CODE HERE ***

def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    z_1, output, _ = forward_prop_func(data, labels, params)
    gradients = {}

    dscores = output - labels
    dscores = np.true_divide(dscores, float(labels.shape[0]))
    dCE = dscores

    gradients['W2'] = np.dot(z_1.T, dCE)
    gradients['W2'] = gradients['W2'] + 2.0*float(reg)*params['W2']
    gradients['b2'] = np.sum(dCE, axis=0, keepdims=True)

    dhidden = np.dot(dCE, params['W2'].T)
    dhidden = np.multiply(dhidden, z_1*(1.0 - z_1))

    gradients['W1'] = np.dot(data.T, dhidden)
    gradients['W1'] = gradients['W1'] + 2.0*float(reg)*params['W1']
    gradients['b1'] = np.sum(dhidden, axis=0, keepdims=True)
    return gradients
    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """
    # *** START CODE HERE ***
    iteration = int(train_data.shape[0]/batch_size)
    for i in range(iteration):
        gradients = backward_prop_func(train_data[batch_size*i:batch_size*(i+1)], train_labels[batch_size*i:batch_size*(i+1)], params, forward_prop_func)
        params['W1'] = params['W1'] - float(learning_rate)*gradients['W1'].reshape(params['W1'].shape)
        params['b1'] = params['b1'] - float(learning_rate)*gradients['b1'].reshape(params['b1'].shape)
        params['W2'] = params['W2'] - float(learning_rate)*gradients['W2'].reshape(params['W2'].shape)
        params['b2'] = params['b2'] - float(learning_rate)*gradients['b2'].reshape(params['b2'].shape)
    # *** END CODE HERE ***

    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, 
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file, max_rows=None):
    if max_rows is None:
        x = np.loadtxt(images_file, delimiter=',')
        y = np.loadtxt(labels_file, delimiter=',')
    else:
        x = np.loadtxt(images_file, delimiter=',', max_rows = max_rows)
        y = np.loadtxt(labels_file, delimiter=',', max_rows = max_rows)
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True, test_set = False):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    if test_set:
        accuracy = nn_test(all_data['test'], all_labels['test'], params)
        print('For model %s, achieved test set accuracy: %f' % (name, accuracy))

def main(num_epochs = 30, plot=True, train_baseline = True, train_regularized=True, test_set = False):

    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:10000,:]
    dev_labels = train_labels[0:10000,:]
    train_data = train_data[10000:,:]
    train_labels = train_labels[10000:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    
    if train_baseline:
        run_train_test('baseline', all_data, all_labels, backward_prop, num_epochs, plot, test_set = test_set)
    if train_regularized:
        run_train_test('regularized', all_data, all_labels, 
            lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
            num_epochs, plot, test_set = test_set)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    main(num_epochs = args.num_epochs)
