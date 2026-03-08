from linear_algebra import vectors
from linear_algebra import matrix

def mean(m):
    return vectors.vsum(m)/len(m)
def variance(v):
    return (1/len(v))*vectors.squared_magnitude(vectors.subtract(v,len(v)*[mean(v)]))
def standard_deviation(v):
    return variance(v)**0.5
def standardize(v):
    s=standard_deviation(v)
    if s==0:
        return len(v)*[0]
    return vectors.scaler_product(1/s,vectors.subtract(v,vectors.scaler_product(mean(v),len(v)*[1])))
def covariance(x,y):
    return (vectors.dotproduct(x,y)/len(x)) - (mean(x)*mean(y))
def covariance_matrix(m):
    return matrix.covarience_matrix(m)
def standardize_matrix(m):
    return matrix.transpose([standardize(i) for i in matrix.transpose(m)])
