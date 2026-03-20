import maths.linear_algebra.vectors as v
import maths.linear_algebra.matrix as mtx
import maths.functions.activations as act
import optimizers.gradient_descent

def binary_const_alpha_sgd(output_vector,input_matrix,alpha=0.01,epoch=100):
    samples,features=mtx.shape(input_matrix)
    theta=[0]*(features+1)
    z=[[1]+ _ for _ in input_matrix]
    #sgd
    for _ in range(epoch):
        for i in  range(samples):
            gradient=v.scaler_product((act.sigmoid(v.dotproduct(theta,z[i]))-output_vector[i]),z[i])
            theta=optimizers.gradient_descent.gradient_step(gradient,theta,alpha)
    return theta

def binary_batch_const_alpha(output_vector,input_matrix,alpha,batch_size=500,epoch=100):
    samples,features=mtx.shape(input_matrix)
    theta=[0]*(features+1)
    z=[[1]+ _ for _ in input_matrix]
    for _ in range(epoch):
        temp=[0]*(features+1)
        for i in range(samples):
            gradient=v.scaler_product((act.sigmoid(v.dotproduct(theta,z[i]))-output_vector[i]),z[i])
            temp=v.add(temp,gradient)
            if((i+1)%batch_size==0):
                theta=v.subtract(theta,v.scaler_product(alpha/batch_size,temp))
                temp=[0]*(features+1)
        if(samples%batch_size!=0):
            theta=v.subtract(theta,v.scaler_product(alpha/(samples%batch_size),temp))
    return theta


