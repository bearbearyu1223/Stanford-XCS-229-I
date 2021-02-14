import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

COLORS = ['red', 'green', 'blue', 'orange']
DATA_DIR = '.'
MAX_LABELED = 5   # Max number of points to label per cluster
UNLABELED = -1    # Cluster label for unlabeled data points


def generate_gaussian(num_examples=100):
    """Generate dataset where each example is sampled from 1 of 4 bivariate Gaussians.
    """

    # Set parameters for each Gaussian in the mixture
    gaussians = [
        ([0.0, .00], .02 * np.eye(2), 200),
        ([.35, .55], .03 * np.eye(2), 200),
        ([0.0, 1.2], .04 * np.eye(2), 200),
        ([-1., 1.4], 1.0 * np.eye(2), 400),
    ]

    # Generate dataset
    examples = []
    for class_idx, (mu, sigma, count) in enumerate(gaussians):
        # Sample class from Gaussian
        class_examples = np.random.multivariate_normal(mu, sigma, count)

        # Add each example to the list
        n_labeled = 0
        for x in class_examples:
            x_dict = {'x_{}'.format(i+1): x_i for i, x_i in enumerate(x)}

            # Only label MAX_LABELED per class
            if n_labeled < MAX_LABELED:
                x_dict['z'] = class_idx
                n_labeled += 1
            else:
                x_dict['z'] = UNLABELED

            examples.append(x_dict)

    random.shuffle(examples)

    df = pd.DataFrame(examples)

    return df


def plot_dataset(df, output_path):
    """Plot a 2D dataset and write to output_path."""
    x = np.array([[row['x_1'], row['x_2']] for _, row in df.iterrows()])
    z = np.array([row['z'] for _, row in df.iterrows()])

    plt.figure(figsize=(12, 8))
    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)
    plt.savefig(output_path)


if __name__ == '__main__':
    np.random.seed(229)
    random.seed(229)

    for split, n in [('train', 1000), ('valid', 44), ('test', 48)]:
        gaussian_df = generate_gaussian(num_examples=n)
        gaussian_df.to_csv(os.path.join(DATA_DIR, '{}.csv'.format(split)), index=False)
        if split == 'train':
            plot_dataset(gaussian_df, os.path.join(DATA_DIR, 'plot.eps'))
