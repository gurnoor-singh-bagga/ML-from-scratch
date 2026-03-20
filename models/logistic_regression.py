import maths.linear_algebra.vectors as v
import maths.linear_algebra.matrix as mtx
import maths.functions.activations as act
from optimizers import gradient_descent
from optimizers import trainers

def logistic_gradient(y,x,theta):
    return v.scaler_product((act.sigmoid(v.dotproduct(theta,x))-y),x)

def binary_const_alpha_sgd(output_vector,input_matrix,alpha=0.01,epoch=100):
    return trainers.sgd_alpha_const(output_vector,input_matrix,logistic_gradient,alpha,epoch)

def binary_batch_const_alpha(output_vector,input_matrix,alpha=0.01,batch_size=500,epoch=100):
    return trainers.batch_const_alpha(output_vector,input_matrix,logistic_gradient,alpha,batch_size,epoch)


