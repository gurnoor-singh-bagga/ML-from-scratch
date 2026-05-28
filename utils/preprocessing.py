import random
import maths.probability_statistics.statistics as stats 
import utils.base as base
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
        print(f"the seed for median-stochastic is {seed}")
    else:
        raise ValueError(
            "choose from DROP, MEAN, MEDIAN, MEDIAN_STOCHASTIC"
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
        if value not in mapping:
            raise ValueError(f"the value of {value} is not mapped")
        row[i]=mapping[value]
    return newx
    