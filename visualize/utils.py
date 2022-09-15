import math

def get_combination(ind, n_samples, n_pairs=None):
    '''
    Given the index of a combination of 2 items, returns the indices of the 2 items
    I.e. given ind in range [0, nC2), returns tuple of 2 indices in range [0, n)
    '''
    if n_pairs is None:
        n_pairs = math.comb(n_samples, 2)
    assert ind >= 0 and ind < n_pairs

    p1 = 0
    sub = n_samples - 1
    while ind - sub >= 0:
        ind -= sub
        p1 += 1
        sub -= 1
    p2 = p1 + 1 + ind

    return p1, p2