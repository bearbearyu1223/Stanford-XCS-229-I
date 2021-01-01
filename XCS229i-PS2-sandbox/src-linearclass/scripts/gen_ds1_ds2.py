import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

DATA_DIR = '.'
np.random.seed(229)


def generate_gaussian(num_examples=100, exponentiate=False):
    """Generate dataset where each class is sampled from a bivariate Gaussian."""

    # Set parameters for each class
    class_labels = (0., 1.)
    mus = ([0.5, 0.5], [2, 2])
    sigmas = ([[1, 0.75], [0.75, 1]], [[1, 0.75], [0.75, 1]])

    # Generate dataset
    examples = []
    for class_label, mu, sigma in zip(class_labels, mus, sigmas):
        # Sample class from Gaussian
        class_examples = np.random.multivariate_normal(mu, sigma, num_examples // len(class_labels))

        # Add each example to the list
        for x in class_examples:
            x_dict = {f'x_{i+1}': x_i for i, x_i in enumerate(x)}
            x_dict['y'] = class_label

            if exponentiate:
                x_dict['x_2'] = math.exp(x_dict['x_2'])
            examples.append(x_dict)

    df = pd.DataFrame(examples)

    return df


def plot_dataset(df, output_path):
    """Plot a 2D dataset and write to output_path."""
    xs = np.array([[row['x_1'], row['x_2']] for _, row in df.iterrows()])
    ys = np.array([row['y'] for _, row in df.iterrows()])

    plt.figure(figsize=(12, 8))
    for x_1, x_2, y in zip(xs[:, 0], xs[:, 1], ys):
        marker = 'x' if y > 0 else 'o'
        color = 'red' if y > 0 else 'blue'
        plt.scatter(x_1, x_2, marker=marker, c=color, alpha=.5)
    plt.savefig(output_path)


if __name__ == '__main__':

    for split, n in [('train', 800), ('valid', 100), ('test', 101)]:
        non_gaussian_df = generate_gaussian(num_examples=n, exponentiate=True)
        non_gaussian_df.to_csv(os.path.join(DATA_DIR, f'ds1_{split}.csv'), index=False)
        if split == 'train':
            plot_dataset(non_gaussian_df, os.path.join(DATA_DIR, 'plot1.png'))

    # ds2 is Gaussian
    for split, n in [('train', 800), ('valid', 100), ('test', 101)]:
        gaussian_df = generate_gaussian(num_examples=n, exponentiate=False)
        gaussian_df.to_csv(os.path.join(DATA_DIR, f'ds2_{split}.csv'), index=False)
        if split == 'train':
            plot_dataset(gaussian_df, os.path.join(DATA_DIR, 'plot2.png'))
