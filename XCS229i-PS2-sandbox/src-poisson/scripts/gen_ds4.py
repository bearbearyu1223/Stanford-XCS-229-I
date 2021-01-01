"""Script to generate ds4 for Poisson regression."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

INCLUDE_TEST_LABELS = False
DATA_DIR = './'
np.random.seed(229)


def generate_poisson(theta, bias, num_examples=500, include_true_labels=True):
    """Generate dataset where each example is drawn form a Poisson distribution.

    The Poisson's natural parameter is a linear combination of features
    (is_sunday, ..., is_saturday, is_nice_weather, num_ads, num_parent_visitors),
    where the linear coefficients are given by theta.
    """
    examples = []
    # Generate randomly selected features and assign to a split
    for _ in range(num_examples):
        # Get input features
        x = get_random_x(theta.shape[0])
        x_dict = {f'x_{i+1}': x_i for i, x_i in enumerate(x)}

        # Get label
        if not include_true_labels:
            # No labels for test set
            x_dict['y'] = -1
        else:
            # Sample label from Poisson(lam)
            lam = np.exp(x.dot(theta) + bias)
            x_dict['y'] = np.random.poisson(lam)
        examples.append(x_dict)

    df = pd.DataFrame(examples)

    return df


def get_random_x(n):
    """Get a random x data point. The returned x should be projected
    to get a log-expected value. Then the resulting Poisson distribution
    should be sampled to get a y-value.
    """
    x = np.zeros(n, dtype=np.float32)

    # Sample day of the week
    day_of_week = np.random.randint(0, 7)
    if day_of_week < 2:
        # Weekend
        x[0] = 1.
    else:
        x[1] = 1.

    # Sample ad budget spending
    x[2] = np.random.random() * 3

    # Sample natural press coverage
    x[3] = np.random.random() * 2

    return x


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

    theta_ = np.array([0.35,   # Weekend
                       0.3,   # Weekday
                       0.28,   # Ads spending (between 0 and 1)
                       0.4])  # Natural press coverage (between 0 and 1)

    bias = 1.1

    # ds4 is drawn from a Poisson distribution
    for split, n in [('train', 2500), ('valid', 250), ('test', 251)]:
        poisson_df = generate_poisson(theta_, bias, num_examples=n)
        print('Generated {} examples in {}'.format(n, split))
        poisson_df.to_csv(os.path.join(DATA_DIR, f'{split}.csv'), index=False)
