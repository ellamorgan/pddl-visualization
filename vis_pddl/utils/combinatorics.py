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


def factoradic_to_permutation(factoradic):
    '''
    Maps a factorial base number (factoradic) to a permutation
    '''
    objects = list(range(len(factoradic)))
    permutation = []

    for ind in factoradic:
        permutation.append(objects[ind])
        del objects[ind]
    
    return permutation


def permutation_to_factoradic(permutation):
    '''
    Maps a permutation to a factorial base number
    '''
    objects = list(range(len(permutation)))
    factoradic = []

    for obj in permutation:
        ind = objects.index(obj)
        factoradic.append(ind)
        del objects[ind]
    
    return factoradic


def factoradic_to_int(factoradic):
    '''
    Maps a factorial base number to a base 10 number
    '''
    result = 0
    factorial = 1
    for i, num in enumerate(factoradic[::-1]):
        result += num * factorial
        factorial *= i + 1
    return result


def int_to_factoradic(num, size):
    '''
    Maps a base 10 number to a factorial base number
    '''
    result = []
    for i in range(size, 0, -1):
        factorial = math.factorial(i - 1)
        rem = math.floor(num / factorial)
        num -= factorial * rem
        result.append(rem)
    return result