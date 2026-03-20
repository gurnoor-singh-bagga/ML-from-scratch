import gradient_descent
from maths.linear_algebra import matrix as mtx
from maths.linear_algebra import vectors as v
from utils import data
#need to update linear regression and logistic regression in future
#need to add suffle before each epoch
def sgd_alpha_const(output_vector,input_matrix,gradient_function,alpha=0.01,epoch=100):
    samples,features=mtx.shape(input_matrix)
    theta=[0]*(features+1)
    z=[[1]+ _ for _ in input_matrix]
    for _ in range(epoch):
        indices=data.shuffle(samples)
        for i in indices:
            gradient=gradient_function(output_vector[i],z[i], theta)
            theta=gradient_descent.gradient_step(gradient,theta,alpha)
    return theta


def batch_const_alpha(output_vector,input_matrix,gradient_function,alpha=0.01,batch_size=500,epoch=100):
    samples,features=mtx.shape(input_matrix)
    theta=[0]*(features+1)
    z=[[1]+ _ for _ in input_matrix]
    for _ in range(epoch):
        temp=[0]*(features+1)
        indices=data.shuffle(samples)
        j=0
        for i in indices:
            j=j+1
            gradient=gradient_function(output_vector[i], z[i],theta)
            temp=v.add(temp,gradient)
            if((j)%batch_size==0):
                j=0
                theta=v.subtract(theta,v.scaler_product(alpha/batch_size,temp))
                temp=[0]*(features+1)
        if(j!=0):
            theta=v.subtract(theta,v.scaler_product(alpha/j,temp))
    return theta
