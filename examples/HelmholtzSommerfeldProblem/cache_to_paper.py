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

reader = csv.DictReader(in_file)

field_names = set()
results = {}
for row in reader:
    input_cols = set(['h', 'kappa'])
    input_data = frozenset({c: row[c] for c in input_cols}.items())
    values = results.setdefault(input_data, {})

    method = row['method'] + ' '
    for col_name in set(row.keys()) - input_cols:
        if col_name == 'method':
            continue
        values[method + col_name] = row[col_name]
        field_names.add(method + col_name)
in_file.close()

field_names = ('h', 'kappa') + tuple(field_names)
writer = csv.DictWriter(out_file, field_names)
writer.writeheader()

for input_, output in results.items():
    row = {**dict(input_), **output}
    writer.writerow(row)

out_file.close()
