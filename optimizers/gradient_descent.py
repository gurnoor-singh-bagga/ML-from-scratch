from maths.linear_algebra import vectors as v
from maths.linear_algebra import matrix as mtx
from maths.calculus import calculus as calc

def gradient_step(gradient, theta,alpha):
    return v.subtract(theta, v.scaler_product(alpha,gradient))