
#vector opretions
def error():
    raise ValueError("invalid input")
def checkdim(a,b):
    if len(a)!=len(b):
        error()
def add(a,b):
    checkdim(a,b)
    return [x+y for x,y in zip(a,b)]
def subtract(a,b):
    checkdim(a,b)
    return[x-y for x,y in zip(a,b)]
def dotproduct(a,b):
    checkdim(a,b)
    c=0
    for x,y in zip(a,b):
        c+=x*y
    return c
def magnitude(a):
    return dotproduct(a,a)**0.5
def scaler_product(s,v):
    return[s*x for x in v]
def distance(a,b):
    return magnitude(subtract(a,b))
def normalize(v):
    m=magnitude(v)
    if m==0:
        error()
    return scaler_product(1/m, v)
def hadamard(a,b):
    checkdim(a,b)
    return [x*y for x,y in zip(a,b)]
def vsum(v):
    s=0
    for i in v:
        s+=i
    return s
def cos_similarity(a,b):
    ma=magnitude(a)
    mb=magnitude(b)
    if ma==0 or mb==0:
        error()
    return(dotproduct(a,b)/(ma*mb))
def squared_magnitude(v):
    return dotproduct(v,v)
def mean(vectors):
    n=len(vectors)
    if n==0:
        error()
    d=len(vectors[0])
    
    result=[0]*d
    
    for v in vectors:
        result=add(result,v)
        
    return scaler_product(1/n,result)
