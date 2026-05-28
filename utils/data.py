#our focus is on creating ml librariries from basic not create my own random or csv reader
import random
import csv

def shuffle(samples):
    indices=list(range(samples))
    random.shuffle(indices)
    return indices

def _tryconvertfloat(value):
    if value=="" or value.strip()=="":
        return None
    try:
        return float(value)
    except ValueError:
        return value
def loadcsv(filepath , targetcol=None):

    x=[]
    y=[]

    with open(filepath,"r") as file:
        reader= csv.reader(file)
        header=next(reader)

        if targetcol==None:
            targetidx=len(header)-1

        elif isinstance(targetcol,str):
            if targetcol not in header:
                raise ValueError(f"colm {targetcol} not found")
            targetidx=header.index(targetcol)

        elif isinstance(targetcol,int):
            if targetcol<0 or targetcol>=len(header):
                raise ValueError("target coloum index out of range")
            targetidx = targetcol

        else: raise ValueError("target colm should be int or str")

        colmname=[col for i,col in enumerate(header) if i!=targetidx]
        for row in reader:
            convertedrow=[_tryconvertfloat(v) for v in row]
            features=[value for i,value in enumerate(convertedrow) if i!=targetidx]
            x.append(features)
            y.append(convertedrow[targetidx])
    return x,y,colmname


def train_test_split(x,y,test_ratio=0.2,seed=None):
    n=len(y)

    if seed==None:
        seed=random.randint(1,9999)
    index=list(range(n))
    split_idx=int(n*(1-test_ratio))
    random.seed(seed)
    random.shuffle(index)
    train=index[:split_idx]
    test=index[split_idx:]
    x_train=[x[i] for i in train]
    y_train=[y[i] for i in train]
    y_test=[y[i] for i in test]
    x_test=[x[i] for i in test]
    print(f"the seed used to split the data is {seed}")
    return x_train,y_train,x_test,y_test, seed