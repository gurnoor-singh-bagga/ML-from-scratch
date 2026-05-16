def constant(alpha, epoch):
    return alpha

def step_decay(drop=0.5, every=10):
    def schedule(alpha, epoch):
        return alpha * (drop ** (epoch // every))
    return schedule

def time_decay(decay=0.01):
    def schedule(alpha, epoch):
        return alpha / (1 + decay * epoch)
    return schedule

def exponential_decay(decay=0.01):
    def schedule(alpha, epoch):
        import math
        return alpha * math.exp(-decay * epoch)
    return schedule