from maths.linear_algebra import vectors as v
from maths.linear_algebra import matrix as mtx
from optimizers import gradient_descent
#for proper scaling this ahve to breaken and loss function is need to build
#improvment can done as it set can be sufflesd for not having any bias based on there order 
def sgd_static_alpha(output_vector,input_matrix,learning_rate=0.01,epoches=100):
    features=len(input_matrix[0])
    samples=len(output_vector)
    theta=[0]*(features+1)
    z = [[1] + row for row in input_matrix]
    #applying sgd
    for _ in range(epoches):
        for i in range(samples):
            gradient=v.scaler_product((output_vector[i],v.dotproduct(theta,z[i])),z[i])
            theta=gradient_descent.gradient_step(gradient, theta,learning_rate)
    return theta
#lu decompostion is needed to impliment in inverse and 
def clossed_form(y,x):
    z= [[1] + row for row in x]
    zt=mtx.transpose(z)
    return mtx.matrix_vector_product((mtx.matrix_matrix_product(mtx.matrix_inverse(mtx.matrix_matrix_product(zt,z)),zt)),y)

def batch(output_vector,input_matrix,alpha,batch_size=500,epoch=100):
    samples,features=mtx.shape(input_matrix)
    theta=[0]*(features+1)
    z=[[1]+ _ for _ in input_matrix]
    for _ in range(epoch):
        temp=[0]*(features+1)
        for i in range(samples):
            gradient=v.scaler_product((-output_vector[i]+v.dotproduct(theta,z[i])),z[i])
            temp=v.add(temp,gradient)
            if((i+1)%batch_size==0):
                theta=v.subtract(theta,v.scaler_product(alpha/batch_size,temp))
                temp=[0]*(features+1)
        if(samples%batch_size!=0):
            theta=v.subtract(theta,v.scaler_product(alpha/(samples%batch_size),temp))
    return theta
