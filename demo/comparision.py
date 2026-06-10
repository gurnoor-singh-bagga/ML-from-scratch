#import
import sys
sys.path.append('.')   # ensures imports work from repo root
from utils import base
from utils.data import loadcsv, train_test_split
from utils import preprocessing  
from models.linear_regression import linearRegression
from models.logistic_regression import logisticRegression
from models.base import TrainMethod
from optimizers import schedules
from utils import metrics
from maths.functions import loss
from sklearn.linear_model import LinearRegression, LogisticRegression
import os
import datetime
import time
#helper functions

class Logger:
    def __init__(self, filepath):
        self.file = open(filepath, 'a')
    
    def log(self, text=''):
        print(text)           # prints to terminal
        self.file.write(text + '\n')  # writes to file simultaneously
    
    def close(self):
        self.file.close()

def print_table(logger, title, results):
    methods = [name for name, _ in results]
    metrics_keys = list(results[0][1].keys())
    
    col_width = 15
    header = f'  {"Metric":<15}' + ''.join(f'{m:>{col_width}}' for m in methods)
    
    logger.log('=' * (15 + col_width * len(methods) + 2))
    logger.log(f'  {title}')
    logger.log('=' * (15 + col_width * len(methods) + 2))
    logger.log(header)
    logger.log('  ' + '-' * (13 + col_width * len(methods)))
    
    for key in metrics_keys:
        row = f'  {key:<15}'
        for _, metrics_dict in results:
            row += f'{metrics_dict[key]:>{col_width}.4f}'
        logger.log(row)
    
    logger.log('=' * (15 + col_width * len(methods) + 2))
def metric_logistic_reg(y_test,preds,probas):
    metrics_project={"Accuracy": metrics.accuracy(y_test, preds),
      "Precision":metrics.precision(y_test, preds),
      "Recall":   metrics.recall(y_test, preds),
      "F1 Score": metrics.f1_score(y_test, preds),
      "BCE Loss": loss.binary_cross_entropy(y_test, probas),
      "Specificity": metrics.specificity(y_test, preds),}
    return metrics_project


def metric_linear_reg(y_test,pred):
    result={ "R2":   metrics.r2_score(y_test,pred),
             "MSE":  loss.mse(y_test,pred),
             "RMSE": loss.rmse(y_test,pred),
             "MAE":  loss.mae(y_test,pred),
             "Adj R2": metrics.adjusted_r2(y_test, pred, 8)}  # 8 features in california 
    return result


def run_titanic(logger):
    x,y,col=loadcsv("data/titanic.csv",targetcol="Survived")
    newx,newcol=preprocessing.drop_colm(x, col, ['PassengerId', 'Name', 'Ticket', 'Cabin'])
    logger.log('colm dropped \'passengerId\',\'Name\', \'Ticket\', \'Cabin\'')
    newx,newy,seed=preprocessing.fill_missing(newx,y,newcol,["Age"],strategy=base.FillStrategy.MEDIAN_STOCHASTIC)
    logger.log(f'seed used for filling miss using median-stochastic:{seed}')
    newx,newy,seed=preprocessing.fill_missing(newx,newy,newcol,["Embarked"],strategy=base.FillStrategy.DROP)
    sexmapping={'male': 0, 'female': 1}
    embarkedmapping={'S': 0, 'C': 1, 'Q': 2}
    logger.log('mapping for catagorial values:\n \'Sex\':{\'male\': 0, \'female\': 1}\n \'Embarked\':{\'S\': 0, \'C\': 1, \'Q\': 2}')
    newx=preprocessing.encode(newx,newcol,'Sex',sexmapping)
    newx=preprocessing.encode(newx,newcol,'Embarked',embarkedmapping)
    x_train,y_train,x_test,y_test,seed=train_test_split(newx,newy,test_ratio=0.2)
    logger.log(f'train test split ratio is 0.2 with seed:{seed}')
    x_train,params=preprocessing.min_max_scaler(x_train,newcol,['Age', 'Fare', 'Pclass', 'SibSp', 'Parch'])
    logger.log('features scaled : Age, Fare, Pclass, SibSp, Parch')    
    x_test=preprocessing.apply_min_max_scaler(x_test,params)
    model_sgd=logisticRegression(method=TrainMethod.SGD,alpha=0.1,epoch=300)
    start = time.time()
    model_sgd.fit(y_train,x_train)
    ####
    preds=model_sgd.predict(x_test)
    probas=model_sgd.predict_proba(x_test)
    metrics_project=metric_logistic_reg(y_test,preds,probas)
    sklearn_logistic=LogisticRegression(max_iter=1000)
    start = time.time()
    sklearn_logistic.fit(x_train,y_train)
    ######
    sklearn_preds = sklearn_logistic.predict(x_test)
    sklearn_probas = sklearn_logistic.predict_proba(x_test)
    sklearn_probas = [row[1] for row in sklearn_probas]
    metrics_sklearn=metric_logistic_reg(y_test,sklearn_preds,sklearn_probas)
    results=[('project' , metrics_project),("sklearn", metrics_sklearn)]
    logger.log('\n' + '━' * 79)
    logger.log(f'  DATASET: Titanic  |  rows: {len(newx)}  |  features: {len(newcol)}  |  task: Classification')
    logger.log('━' * 79)
    print_table(logger,"Logistic Regression", results )

def run_california(logger):
    x,y,col=loadcsv("data/california_housing.csv")
    x_train,y_train,x_test,y_test,seed=train_test_split(x,y,test_ratio=0.2)
    logger.log(f'train test split ratio is 0.2 with seed:{seed}')
    x_train,params=preprocessing.standardize(x_train,col)
    x_test=preprocessing.apply_standardize(x_test,params)
    y_mean = sum(y_train) / len(y_train)
    y_std = (sum((v - y_mean)**2 for v in y_train) / len(y_train))**0.5
    y_train = [(v - y_mean) / y_std for v in y_train]
    y_test  = [(v - y_mean) / y_std for v in y_test]
    project_model_sgd=linearRegression(method=TrainMethod.SGD,alpha=0.00001,epoch=300,schedule=schedules.step_decay(0.5,100))
    start = time.time()
    project_model_sgd.fit(y_train,x_train)
    logger.log(f'  project sgd training time: {time.time() - start:.3f}s')
    sgd_pred=project_model_sgd.predict(x_test)
    sgd_metric=metric_linear_reg(y_test,sgd_pred)
    project_model_batch=linearRegression(method=TrainMethod.BATCH,alpha=0.01,batch_size=32,epoch=300,schedule=schedules.step_decay(0.5,100))
    start = time.time()
    project_model_batch.fit(y_train,x_train)
    logger.log(f'  project batch training time: {time.time() - start:.3f}s')
    batch_pred=project_model_batch.predict(x_test)
    batch_metric=metric_linear_reg(y_test,batch_pred)
    project_model_closed=linearRegression(method=TrainMethod.CLOSED_FORM)
    start = time.time()
    project_model_closed.fit(y_train,x_train)
    logger.log(f'  closed training time: {time.time() - start:.3f}s')
    closed_pred=project_model_closed.predict(x_test)
    closed_metric=metric_linear_reg(y_test,closed_pred)
    sklearn_model=LinearRegression()
    start = time.time()
    sklearn_model.fit(x_train,y_train)
    logger.log(f'  sklearn training time: {time.time() - start:.3f}s')
    sklearn_pred=sklearn_model.predict(x_test)
    sklearn_metric=metric_linear_reg(y_test,sklearn_pred)
    results=[("project_sgd",sgd_metric),("project_batch",batch_metric),
             ("project_closed",closed_metric),("sklearn",sklearn_metric)]
    logger.log('\n' + '━' * 79)
    logger.log(f'  DATASET: Califonia-Housing  |  rows: {len(x)}  |  features: {len(col)}  |  task:Regression. ')
    logger.log('━' * 79)
    print_table(logger,"Linear Regression",results)



if __name__ == '__main__':
    logger = Logger('demo/results.md')
    logger.log('\n' + '#' * 79)
    logger.log(f'  RUN: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.log('#' * 79)
    logger.log("Note:1.project works on list not on arrays so a large diif in time will be seen" \
    "\n     2.there isnt regulerisation in so the this may overfit the things espesially in logistic regression ")
    run_titanic(logger)
    run_california(logger)
    logger.close()

