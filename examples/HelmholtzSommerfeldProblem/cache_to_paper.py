"""
This "flattens" the data to trials that depend on (kappa, h)
instead of (kappa, h, method) by changing output_name to
method + ' ' + output_name,

e.g. 'L2 Error' -> 'pml L2 Error'

This makes it easier to print in the paper
"""


import csv

in_file = open('data/circle_in_square.csv')
out_file = open('2d_data.csv', 'w')
input_cols = set(['h', 'kappa', 'pyamg_maxiter'])
# map output col to its dependencies in addition to *input_cols*
output_cols = {'L2 Error': ['method', 'pc_type'],
               'H1 Error': ['method', 'pc_type'],
               'Iteration Number': ['method', 'pc_type'],
               }

def get_output(row):
    output = {}
    for col, deps in output_cols.items():
        col_name = " ".join([row[dep] for dep in deps] + [col])
        output[col_name] = row[col]
    return output

reader = csv.DictReader(in_file)

field_names = set()
results = {}
for row in reader:
    input_data = frozenset({c: row[c] for c in input_cols}.items())
    values = results.setdefault(input_data, {})
    values.update(get_output(row))
    field_names |= set(values.keys())

in_file.close()

field_names = tuple(input_cols) + tuple(field_names)
writer = csv.DictWriter(out_file, field_names)
writer.writeheader()

def get_key(items):
    input_, output = items
    dict_ = dict(input_)
    key = tuple([dict_[c] for c in input_cols])
    return key


prev_kappa = None
for input_, output in sorted(results.items(), key=get_key):
    input_ = dict(input_)
    row = {**input_, **output}
    for col in field_names:
        if col not in row or row[col] == '':
            row[col] = 'nan'
    writer.writerow(row)

out_file.close()
