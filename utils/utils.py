def atleast_kdim(x, ndim):
    shape = x.shape + (1,) * (ndim - len(x.shape))
    return x.reshape(shape)