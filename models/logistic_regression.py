import maths.linear_algebra.vectors as v
import maths.linear_algebra.matrix as mtx
import maths.functions.activations as act
import optimizers.trainers as trainers
import models.base as base
import utils.metrics as metrics

def logistic_gradient(y,x,theta):
    return v.scaler_product((act.sigmoid(v.dotproduct(theta,x))-y),x)

class logisticRegression:
    def __init__(self,method=base.TrainMethod.SGD,alpha=0.01,batch_size=500,epoch=100,schedule=None):
        self.method=method
        self.alpha=alpha 
        self.batch_size=batch_size
        self.epoch=epoch
        self.schedule=schedule
        self.theta=None

    def fit( self,y,X):
        if  self.method==base.TrainMethod.SGD:
            self.theta=trainers.sgd(y,X,logistic_gradient,self.alpha,self.epoch,self.schedule)
        elif self.method==base.TrainMethod.BATCH:
             self.theta = trainers.batch(y,X,logistic_gradient, self.alpha,self.batch_size,self.epoch,self.schedule)
        elif self.method ==base.TrainMethod.CLOSED_FORM:
            raise ValueError("logistic regression doesnt have closed form use TrainMethod.SGD or TrainMeathod.BATCH")
        return self
    def predict_proba(self,X):
        if self.theta==None:
            raise RuntimeError("call fit() before predict()")
        z=[[1]+row for row in X]
        return [act.sigmod(v.dotproduct(self.theta,row)) for row in z]

    def predict(self,X):
        return [1 if p>=0.5 else 0  for p in self.predict_proba(X)]
    def score(self,y,X):
        y_pred=self.predict(X)
        return metrics.accuracy(y,y_pred)



