#our focus is on creating ml librariries from basic not create my own random or exp
import random

def shuffle(samples):
    indices=list(range(samples))
    random.shuffle(indices)
    return indices