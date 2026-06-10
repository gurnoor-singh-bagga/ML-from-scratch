import random
import maths.probability_statistics.statistics as stats 
import utils.base as base
import maths.linear_algebra.vectors as v
import maths.linear_algebra.matrix as mtx

def drop_colm(x,colm_names,colm_to_drop):
    index=set()
    for colm in colm_to_drop:
        try:
            index.add(colm_names.index(colm))
        except ValueError:
            print(f"colm {colm} not exist")
    newx=[]
    new_colm_names=[col for i,col in enumerate(colm_names) if i not in index]
    for row in x:
        newrow=[v for i, v in enumerate(row) if i not in index ]
        newx.append(newrow)
    return newx,new_colm_names
#need to break this function in smaller chunks
def fill_missing(x,y,colm_names,colm,strategy=base.FillStrategy.MEDIAN,seed=None):
    index = set()
    for i in colm:
        try:
            index.add(colm_names.index(i))
        except ValueError:
            print(f"colm {i} not exist")
    newx = [row[:] for row in x]
    newy = y.copy()

    if strategy == base.FillStrategy.DROP:
        for i in index:
            filtered = [
                (row, target)
                for row, target in zip(newx, newy)
                if row[i] is not None
            ]
            if filtered:
                newx, newy = zip(*filtered)
                newx = list(newx)
                newy = list(newy)
            else:
                newx, newy = [], []
    elif strategy == base.FillStrategy.MEAN:
        for i in index:
            x2 = [row[i] for row in x if row[i] is not None]
            if not x2:
                continue
            mean = stats.mean(x2)
            for row in newx:
                if row[i] is None:
                    row[i] = mean
    elif strategy == base.FillStrategy.MEDIAN:
        for i in index:
            x2 = [row[i] for row in x if row[i] is not None]
            if not x2:
                continue
            median = stats.median(x2)
            for row in newx:
                if row[i] is None:
                    row[i] = median
    elif strategy == base.FillStrategy.MEDIAN_STOCHASTIC:
        if seed is None:
            seed = random.randint(1, 9999)
            print(f"the seed for median-stochastic is {seed}")
        random.seed(seed)
        for i in index:
            x2 = [row[i] for row in x if row[i] is not None]
            if not x2:
                continue
            median = stats.median(x2)
            std = stats.standard_deviation(x2)
            col_min = min(x2)
            col_max = max(x2)
            for row in newx:
                if row[i] is None:
                    value = random.gauss(median, std)
                    row[i] = max(col_min, min(col_max, value))
        
    elif strategy == base.FillStrategy.MODE:
        for i in index:
            x2 = [row[i] for row in x if row[i] is not None]
            if not x2:
                continue
            mode=stats.mode(x2)
            for row in newx:
                if row[i] is None:
                    row[i] = mode
    elif strategy== base.FillStrategy.MODE_STOCHASTIC:
        if seed is None:
                seed =random.randint(1,9999)
                print(f"the seed for mode-stochastic is {seed}")
        random.seed(seed)
        for i in index:
            freq={}
            for row in newx:
                if row[i] is None:
                    continue
                elif row[i] in freq:
                    freq[row[i]]+=1
                else:
                    freq[row[i]]=1
            item=list(freq.keys())
            ratio=list(freq.values())
            for row in newx:
                if row[i] is None:
                    row[i]=random.choices(item,weights=ratio,k=1)[0]

    else:
        raise ValueError(
            "choose from DROP, MEAN, MEDIAN, MEDIAN_STOCHASTIC, MODE , MODE_STOCHASTIC"
        )

    return newx, newy, seed

def encode(x,colm_name,colm,mapping):
    try:
        i=colm_name.index(colm)
    except ValueError:
        raise ValueError(f"colm {colm} not found")
    newx=[row[:] for row in x]
    for row in newx:
        value=row[i]
        if value is None:
            raise ValueError(f"column '{colm}' has missing values — fill them before encoding")
        if value not in mapping:
            raise ValueError(f"the value of {value} is not mapped")
        row[i]=mapping[value]
    return newx

def min_max_scaler(x, colm_name,col=None):
    if  col is None:
        index=set(range(len(colm_name)))
    else:
        index=set()
        for colm in col:
            try:
                index.add(colm_name.index(colm))
            except ValueError:
                print(f"colm {colm} not exist")
    newx = [row[:] for row in x]
    params=[None]*len(x[0])
    for i in index:
        x2=mtx.get_col(x,i+1)
        cmax=max(x2)
        cmin=min(x2)
        if cmax== cmin:
            continue
        else:
            x2=v.scaler_product(1/(cmax-cmin),v.subtract(x2,[cmin]*len(x2)))
            for j, row in enumerate(newx):
                row[i]=x2[j]
            params[i]={"min":cmin,"max":cmax}
    return newx, params

def apply_min_max_scaler(x, params):
    newx=[row[:] for row in x]
    for i, para in enumerate(params):
        if para is None:
            continue
        else:
            x2=mtx.get_col(x,i+1)
            cmin=para["min"]
            cmax=para["max"]
            x2=v.scaler_product(1/(cmax-cmin),v.subtract(x2,[cmin]*len(x2)))
            for j, row in enumerate(newx):
                row[i]=x2[j]
    return newx
   
def standardize(x, colm_name,col=None):
    if  col is None:
        index=set(range(len(colm_name)))
    else:
        index=set()
        for colm in col:
            try:
                index.add(colm_name.index(colm))
            except ValueError:
                print(f"colm {colm} not exist")
    newx = [row[:] for row in x]
    params=[None]*len(x[0])
    for i in index:
        x2=mtx.get_col(x,i+1)
        mean=stats.mean(x2)
        std=stats.standard_deviation(x2)
        x2=stats.standardize(x2)
        for j, row in enumerate(newx):
            row[i]=x2[j]
        params[i]={"mean":mean,"std":std}
    return newx, params

def apply_standardize(x,params):
    newx=[row[:] for row in x]
    for i, para in enumerate(params):
        if para is None:
            continue
        else:
            x2=mtx.get_col(x,i+1)
            mean=para["mean"]
            std=para["std"]
            x2=v.scaler_product(1/(std),v.subtract(x2,[mean]*len(x2)))
            for j, row in enumerate(newx):
                row[i]=x2[j]
    return newx
