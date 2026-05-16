import math
import math.linear_algebra.vectors as v

def mse(y_true , y_pred):
    #mean of sqr diff
    diff= v.subtract(y_true,y_pred)
    return v.squared_magnitude(diff)/len(y_true)

def rmse(y_true, y_pred):
    return mse(y_true, y_pred)**0.5

def mae(y_true ,y_pred):
    return sum(abs (a-b) for a , b in zip(y_true, y_pred))/len(y_true)

def binary_cross_entropy(y_true, y_pred):
    eps=1e-15
    loss=0
    for y, p in zip(y_true , y_pred):
        p=max(eps, min(1-eps,p))
        loss+=-(y*math.log(p) +(1-y)*(math.log(1-p)))
    return loss/len(y_true)

