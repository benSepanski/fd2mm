import pickle

file_dir = "tex_files/"
file_name = "cached_results.pickle"

pickle_in = open(file_dir + file_name, 'rb')
known_results = pickle.load(pickle_in)
pickle_in.close()

new_results = {}

# {{{ Alter cached results:
for result in known_results:
    """
    if 'pml(quadratic)' not in result:
        new_results[result] = known_results[result]
    """
# }}}

pickle_out = open(file_dir + file_name, 'wb')
pickle.dump(new_results, pickle_out)
pickle_out.close()
