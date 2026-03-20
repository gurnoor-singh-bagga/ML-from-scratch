import gradient_descent
from maths.linear_algebra import matrix as mtx
from maths.linear_algebra import vectors as v
#need to update linear regression and logistic regression in future
#need to add suffle before each epoch
def sgd_alpha_const(output_vector,input_matrix,gradient_function,alpha=0.01,epoch=100):
    samples,features=mtx.shape(input_matrix)
    theta=[0]*(features+1)
    z=[[1]+ _ for _ in input_matrix]
    for _ in range(epoch):
        for i in range (samples):
            gradient=gradient_function(output_vector,z, theta,i)
            theta=gradient_descent.gradient_step(gradient,theta,alpha)
    return theta


def batch_const_alpha(output_vector,input_matrix,gradient_function,alpha=0.01,batch_size=500,epoch=100):
    samples,features=mtx.shape(input_matrix)
    theta=[0]*(features+1)
    z=[[1]+ _ for _ in input_matrix]
    for _ in range(epoch):
        temp=[0]*(features+1)
        for i in range(samples):
            gradient=gradient_function(output_vector, z,gradient_function, i)
            temp=v.add(temp,gradient)
            if((i+1)%batch_size==0):
                theta=v.subtract(theta,v.scaler_product(alpha/batch_size,temp))
                temp=[0]*(features+1)
        if(samples%batch_size!=0):
            theta=v.subtract(theta,v.scaler_product(alpha/(samples%batch_size),temp))
    return theta
