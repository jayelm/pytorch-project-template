"""
Generate simple binary classification data
"""

import numpy as np


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--w', type=int, default=3, help='W term')
    parser.add_argument('--b', type=int, default=2, help='B term')
    parser.add_argument('--noise', type=float, default=0.5, help='Gaussian noise added to (post-sigmoid) output')
    parser.add_argument('--n', type=int, default=5000, help='Amount to generate')
    parser.add_argument('--min_x', type=int, default=-5, help='Minimum x value')
    parser.add_argument('--max_x', type=int, default=5, help='Maximum x value')

    args = parser.parse_args()

    random = np.random.RandomState(args.seed)

    x = np.linspace(args.min_x, args.max_x, num=args.n)

    transformed_x = (args.w * x) + args.b
    py = 1 / (1 + np.exp(-transformed_x))
    py_noised = py + random.normal(0, args.noise, size=x.shape)

    y = (py_noised > 0.5) * 1

    xandy = ['{},{}'.format(i, j) for i, j in zip(x, y)]
    random.shuffle(xandy)

    with open('data_{}w_{}b.csv'.format(args.w, args.b), 'w') as f:
        f.write('x,y\n')
        f.write('\n'.join(xandy))
        f.write('\n')
