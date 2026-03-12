from linear_algebra import vectors as v
from linear_algebra import matrix as m

def numarical_derivative(f,x,h=1e-5):
    #for single varible funtion
    return (f(x+h)-f(x-h))/(2*h)
def partial_derivative(f,x,i,h=1e-5):
    e=len(x)*[0]
    e[i-1]=1
    return (f(v.add(x,v.scaler_product(h,e)))-f(v.subtract(x,v.scaler_product(h,e))))/h
def gradiant(f,x,h=1e-5):
    n=len(x)
    e=n*[0]
    g=[]
    for i in range(n):
        e[i]=1
        g.append((f(v.add(x,v.scaler_product(h,e)))-f(v.subtract(x,v.scaler_product(h,e))))/h)
        e[i]=0
    return g



