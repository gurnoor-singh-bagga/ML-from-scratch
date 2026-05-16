import maths.linear_algebra.vectors as v

def gradient_step(gradient, theta,alpha):
    return v.subtract(theta, v.scaler_product(alpha,gradient))