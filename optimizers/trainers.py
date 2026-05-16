import optimizers.gradient_descent as gradient_descent
import maths.linear_algebra.matrix as mtx
import maths.linear_algebra.vectors as v
import utils.data as data
import optimizers.schedules as schedules
#need to update linear regression and logistic regression in future
#need to add suffle before each epoch
def sgd(output_vector,input_matrix,gradient_function,alpha=0.01,epoch=100,schedule=None):
    schedule=schedule or schedules.constant
    samples,features=mtx.shape(input_matrix)
    theta=[0]*(features+1)
    z=[[1]+ _ for _ in input_matrix]
    for ep in range(epoch):
        current_alpha=schedule(alpha,ep)
        indices=data.shuffle(samples)
        for i in indices:
            gradient=gradient_function(output_vector[i],z[i], theta)
            theta=gradient_descent.gradient_step(gradient,theta,current_alpha)
    return theta


def batch(output_vector,input_matrix,gradient_function,alpha=0.01,batch_size=500,epoch=100, schedule=None):
    schedule=schedule or schedules.constant
    samples,features=mtx.shape(input_matrix)
    theta=[0]*(features+1)
    z=[[1]+ _ for _ in input_matrix]
    for ep in range(epoch):
        current_alpha=schedule(alpha,ep)
        temp=[0]*(features+1)
        indices=data.shuffle(samples)
        j=0
        for i in indices:
            j=j+1
            gradient=gradient_function(output_vector[i], z[i],theta)
            temp=v.add(temp,gradient)
            if((j)%batch_size==0):
                j=0
                theta=v.subtract(theta,v.scaler_product(current_alpha/batch_size,temp))
                temp=[0]*(features+1)
        if(j!=0):
            theta=v.subtract(theta,v.scaler_product(current_alpha/j,temp))
    return theta
