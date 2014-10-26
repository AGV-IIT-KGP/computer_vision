def greyscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
