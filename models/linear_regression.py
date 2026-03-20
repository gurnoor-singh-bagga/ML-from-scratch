from maths.linear_algebra import vectors as v
from maths.linear_algebra import matrix as mtx
from optimizers import gradient_descent
from optimizers import trainers
#for proper scaling this ahve to breaken and loss function is need to build
#improvment can done as it set can be sufflesd for not having any bias based on there order 

def linear_gradient(y,x,theta,):
    return v.scaler_product((v.dotproduct(theta,x)-y),x)

def sgd_static_alpha(output_vector,input_matrix,learning_rate=0.01,epoch=100):
    return trainers.sgd_alpha_const(output_vector,input_matrix,linear_gradient,learning_rate,epoch)
#lu decompostion is needed to impliment in inverse and x matrix should not have two features with co variance factor 1
def clossed_form(y,x):
    z= [[1] + row for row in x]
    zt=mtx.transpose(z)
    return mtx.matrix_vector_product((mtx.matrix_matrix_product(mtx.matrix_inverse(mtx.matrix_matrix_product(zt,z)),zt)),y)

def batch_const_alpha(output_vector,input_matrix,alpha,batch_size=500,epoch=100):
    return trainers.batch_const_alpha(output_vector,input_matrix,linear_gradient,alpha,batch_size,epoch)
