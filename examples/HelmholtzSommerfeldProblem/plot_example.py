import matplotlib.pyplot as plt
from utils.plotting import make_plot


def test_function(row):
    return row['method'] != 'transmission' and float(row['h']) < 0.06 and float(row['kappa']) <= 10.0


make_plot('data/2d_hankel_trial.csv', 'kappa', 'L2 Relative Error',
          fixed_vars=['h'],
          group_by=['method'],
          nrows=1, ncols=1, test_function=test_function,
          x_type=float, y_type=float)
plt.show()
