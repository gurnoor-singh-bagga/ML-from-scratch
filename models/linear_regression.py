import maths.linear_algebra.vectors as v
import maths.linear_algebra.matrix as mtx
import optimizers.trainers as trainers
import utils.metrics as metrics
import models.base as base

def linear_gradient(y,x,theta,):
    return v.scaler_product((v.dotproduct(theta,x)-y),x)

class linearRegression:
    def __init__(self,method=base.TrainMethod.SGD,alpha=0.01,batch_size=500, epoch=100,schedule=None):
        self.method=method
        self.alpha=alpha
        self.batch_size=batch_size
        self.epoch=epoch
        self.schedule=schedule
        self.theta=None

    def fit( self,y,X):
        if  self.method==base.TrainMethod.SGD:
            self.theta=trainers.sgd(y,X,linear_gradient,self.alpha,self.epoch,self.schedule)
        elif self.method==base.TrainMethod.BATCH:
             self.theta = trainers.batch(y,X,linear_gradient, self.alpha,self.batch_size,self.epoch,self.schedule)
        elif self.method ==base.TrainMethod.CLOSED_FORM:
             z=[[1]+row for row in X]
             zt=mtx.transpose(z)
             self.theta = mtx.matrix_vector_product(mtx.matrix_matrix_product(mtx.matrix_inverse(mtx.matrix_matrix_product(zt, z)), zt), y)
        return self
    def predict(self,X):
        if self.theta==None:
            raise RuntimeError("call fit() before predict()")
        z=[[1] +row for row in X]
        return [v.dotproduct(self.theta,row) for row in z]
    def score(self,y,X):
        y_pred=self.predict(X)
        return metrics.r2_score(y,y_pred)