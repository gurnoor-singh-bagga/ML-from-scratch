from maths.linear_algebra import vectors as v
from optimizers import gradient_descent
#for proper scaling this ahve to breaken and loss function is need to build
#improvment can done as it set can be sufflesd for not having any bias based on there order 
def linear_regression_sgd_static_alpha(y,x,alpha=0.001,iteration=10000):
    n=len(x[0])
    m=len(y)
    theta=[0]*(n+1)
    z = [[1] + row for row in x]
    #applying sgd
    for _ in range(iteration):
        for i in range(m):
            gradient=v.scaler_product((y[i]-v.dotproduct(theta,z[i])),z[i])
            theta=gradient_descent.gradient_step(gradient, theta,alpha)
    return theta
