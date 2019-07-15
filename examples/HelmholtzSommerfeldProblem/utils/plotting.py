import csv
import matplotlib.pyplot as plt


def to_printable(key, val=None):
    try:
        # Catch any integers before formatting as floats
        val = int(val)
    except ValueError:
        try:
            val = float(val)
            if 0.1 < val < 10.0:
                val = '%.2f' % val
            else:
                val = '%.2e' % val
        except ValueError:
            pass
    except TypeError:
        pass

    if key == 'l2_relative_error':
        key = 'L^2 Relative Error'
    elif key == 'h1_relative_error':
        key = 'H^1 Relative Error'
    elif key == 'fmm_order' and val == '':
        return ''

    if val is not None:
        return '%s=%s' % (key, val)
    else:
        return '%s' % key


def make_plot(csv_file_name, x_var, y_var, fixed_vars=None,
              exclude_from_legend=None,
              nrows=2, ncols=2, use_legend=True, test_function=None,
              x_type=None, y_type=None):
    """
        :arg fixed_vars: An iterable of names.
                        Makes new plot when these variables
                        change.
        :arg exclude_from_legend: Extra variables to exclude from the legend
        :arg nrows: number of rows per figure
        :arg ncols: number of columns per figure
        :arg test_function: A function applied to each row
                            which, if it returns *True*, the row will
                            be used, and otherwise it will not
        :arg x_type: Type to cast x values to, or *None* if string
        :arg y_type: Type to cast y values to, or *None* if string
    """
    if exclude_from_legend is None:
        exclude_from_legend = []
    exclude_from_legend = set(exclude_from_legend)

    if test_function is None:
        def test_function(row):
            return True


    # {{{ Read data into variable *data* as a list of dicts

    in_file = open(csv_file_name)
    reader = csv.DictReader(in_file)
    data = [row for row in reader if test_function(row)]
    in_file.close()

    # }}}

    # {{{ Compute number of plots

    if fixed_vars is None:
        fixed_vars = []
    fixed_variables = tuple([fv for fv in fixed_vars])

    # Maps fixed portions to plot data
    data_subsets = {}
    fixed_portion_to_axis = {}

    num_axes = 0
    for row in data:
        fixed_portion = tuple([row[fv] for fv in fixed_variables])
        data_subsets.setdefault(fixed_portion, []).append(row)

        if fixed_portion not in fixed_portion_to_axis:
            if num_axes % (nrows * ncols) == 0:
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

            row_index = (num_axes // nrows) % nrows
            col_index = (num_axes // ncols) % ncols

            if nrows > 1 and ncols > 1:
                fixed_portion_to_axis[fixed_portion] = axes[row_index][col_index]
            elif nrows > 1:
                fixed_portion_to_axis[fixed_portion] = axes[row_index]
            elif ncols > 1:
                fixed_portion_to_axis[fixed_portion] = axes[col_index]
            else:
                fixed_portion_to_axis[fixed_portion] = axes
            num_axes += 1

    for fixed_portion, subset in data_subsets.items():
        # {{{ Group together points in the subset
        group_to_xy = {}

        for row in subset:
            group_keys = {}
            for key, val in row.items():
                if key not in [x_var, y_var] and key not in fixed_variables and \
                        key not in exclude_from_legend:
                    group_keys[key] = val

            group_to_xy.setdefault(frozenset(group_keys.items()), []).append(
                (row[x_var], row[y_var]))

        # }}}

        axis = fixed_portion_to_axis[fixed_portion]
        title = ''
        for key, val in zip(fixed_variables, fixed_portion):
            title += to_printable(key, val) + '; '
        axis.set_title(title)

        label_used = False
        for group, xy_list in group_to_xy.items():
            x, y = zip(*xy_list)

            if x_type is not None:
                x = list(map(x_type, x))
            if y_type is not None:
                y = list(map(y_type, y))

            label = ''
            for key, val in group:
                if key not in exclude_from_legend:
                    label += to_printable(key, val) + '; '

            if not label_used:
                label_used = label != ''

            axis.set_xlabel(to_printable(x_var))
            axis.set_ylabel(to_printable(y_var))
            axis.scatter(x, y, label=label)

        if use_legend and label_used:
            axis.legend()

"""
def test_function(row):
    return row['method'] != 'transmission'

make_plot('data/2d_hankel_trial.csv', 'kappa', 'l2_relative_error',
          fixed_vars=['h'],
          exclude_from_legend=['h1_relative_error', 'ndofs', 'fmm_order', 'degree'],
          nrows=1, ncols=1, test_function=test_function,
          x_type=float, y_type=float)
plt.show()
"""
