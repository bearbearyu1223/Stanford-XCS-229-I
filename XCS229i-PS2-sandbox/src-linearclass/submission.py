import numpy as np
import util


def main_GDA(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)
    x_eval = util.add_intercept(x_eval)

    # Use np.savetxt to save outputs from validation set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('GDA Accuracy: %.2f' % np.mean((yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.theta_0 = theta_0

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples = x.shape[0]
        dim = x.shape[1]
        if self.theta is None:
            self.theta = np.zeros(dim + 1)
        phi = np.true_divide(len(y[y == 1.0]), n_examples)
        mu_0 = np.true_divide(np.sum(x[y == 0.0], axis=0), len(y[y == 0.0]))
        mu_1 = np.true_divide(np.sum(x[y == 1.0], axis=0), len(y[y == 1.0]))
        sigma = np.true_divide((x[y == 1.0] - mu_1).transpose().dot(x[y == 1.0] - mu_1) \
                               + (x[y == 0.0] - mu_0).transpose().dot((x[y == 0.0] - mu_0)), len(x))
        sigma_inv = np.linalg.pinv(sigma)
        theta_0 = 0.5*(np.dot(mu_0, sigma_inv).dot(mu_0) - np.dot(mu_1, sigma_inv).dot(mu_1)) - np.log((1.0 -phi)/phi)
        theta = -np.dot((mu_0 - mu_1).transpose(), sigma_inv)
        self.theta = np.append(theta_0, theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        predictions = []
        for x_i in x:
            theta_x = np.dot(self.theta.transpose(), x_i)
            sigmoid = 1.0/(1.0 + np.exp(-theta_x))
            predictions.append(sigmoid)
        return np.array(predictions)
        # *** END CODE HERE


def main_LogReg(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)

    # Use np.savetxt to save predictions on eval set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('LR Accuracy: %.2f' % np.mean((yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        for _ in range(self.max_iter):
            theta_prev = self.theta
            diagonal = []
            log_likelihood_derivative = np.zeros(x.shape[1])
            for x_i, y_i in zip(x, y):
                g_theta_x_i = 1.0/(1.0 + np.exp(-np.dot(self.theta.transpose(), x_i)))
                g_theta_x_i_derivative = g_theta_x_i * (1.0 - g_theta_x_i)
                diagonal.append(g_theta_x_i_derivative)

                log_likelihood_derivative = log_likelihood_derivative \
                                            + np.true_divide((y_i - g_theta_x_i) * x_i, x.shape[0])
            hessian = np.dot(np.dot(x.transpose(), np.true_divide(np.diag(diagonal), x.shape[0])), x)
            hessian_inverse = np.linalg.pinv(hessian)
            self.theta = self.theta + self.step_size * np.dot(hessian_inverse, log_likelihood_derivative)
            diff = np.linalg.norm((self.theta - theta_prev), ord=1)

            if diff < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        predictions = []
        for x_i in x:
            theta_x = np.dot(self.theta.transpose(), x_i)
            sigmoid = 1.0/(1.0 + np.exp(-theta_x))
            predictions.append(sigmoid)
        return np.array(predictions)
        # *** END CODE HERE ***


if __name__ == '__main__':
    main_LogReg(train_path='ds1_train.csv',
                valid_path='ds1_valid.csv',
                save_path='logreg_pred_1.txt')

    main_LogReg(train_path='ds2_train.csv',
                valid_path='ds2_valid.csv',
                save_path='logreg_pred_2.txt')
    main_GDA(train_path='ds1_train.csv',
             valid_path='ds1_valid.csv',
             save_path='gda_pred_1.txt')

    main_GDA(train_path='ds2_train.csv',
             valid_path='ds2_valid.csv',
             save_path='gda_pred_2.txt')
