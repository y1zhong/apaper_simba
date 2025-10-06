import pickle

def save_pickle(x, filename):
    # mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print('save',filename)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    print(f'Pickle loaded from {filename}')
    return x