#matrix opretion
import vectors

def shape(m):
    return[len(m),len(m[0])]
def get_row(m,i):
    if shape(m)[0]>=i and i>=1:
        return m[i-1]
    vectors.error()
def get_col(m,i):
    if shape(m)[1]>=i and i>=1:
        return [m[x][i-1] for x in range(len(m))]
def add_matrix(a,b):
    if shape(a)==shape(b):
        return[vectors.add(x,y) for x,y in zip(a,b)]
    vectors.error()
def subtract_matrix(a,b):
    if shape(a)==shape(b):
        return[vectors.subtract(x,y) for x,y in zip(a,b)]
    vectors.error()
def scaler_multiply_matrix(s, m):
    return[vectors.scaler_product(s,x) for x in m]
def transpose(m):
    return[get_col(m, i+1) for i in range(len(m[0]))]
def matrix_vector_product(m,v):
    #vXM or mxV ? ->Mv
    return[vectors.dotproduct(v,i) for i in m]
def matrix_matrix_product(a,b):
    return transpose([matrix_vector_product(a,i) for i in transpose(b)])
def zero_matrix(rows,colm):
    return [[colm*[0]] for _ in range(rows)]
def one_matrix(rows,colm):
    return [[colm*[1]] for _ in range(rows)]
def identiy_matrix(n):
    m=zero_matrix(n,n)
    for i in range(n):
        m[i][i]=1
    return m
def center_matrix(m):
    return subtract_matrix(m,len(m)*[vectors.mean(m)])
def covarience_matrix(x):
    xc=center_matrix(x)
    return scaler_multiply_matrix(1/(len(xc)-1),matrix_matrix_product(transpose(xc),xc))
def minor_matrix(m,id ,jd):
    #id, jd are from 1 to n while i or j goes from 0 to n-1
    return[[val for j, val in enumerate(row) if (j+1)!=jd] for i, row in enumerate(m) if (i+1)!=id]
def twodeterminent(m):
    return m[0][0]*m[1][1]-m[0][1]*m[1][0]
#define lu compostion can help in impletion of determinent , matrix inverse , solving linear equation etc
def determinent(m):
    if shape(m)==[2,2]:
        return twodeterminent(m)
    else:
        d=0
        for j, val in enumerate(m[0]):
            d+= ((-1)**j)*val*determinent(minor_matrix(m,1,j+1))
        return d
def matrix_inverse(matrix):
    if determinent(matrix)==0:
        vectors.error()
    m=matrix.copy()
    n=len(m)
    i=identiy_matrix(n)
    for k in range(n):
        while(True):
            c=m[k][k]
            if c==0 and k!=n-1:
                tempm=m[k]
                tempi=i[k]
                for i in range(k,n-1):
                    m[k]=m[k+1]
                    i[k]=i[k+1]
                m[n-1]=tempm
                i[n-1]=tempi
                continue
            break
        i[k]=vectors.scaler_product(1/c,i[k])
        m[k]=vectors.scaler_product(1/c,m[k])
        for s in range(k+1,n):
            i[s]=vectors.subtract(i[s], vectors.scaler_product(m[s][k],i[k]))
            m[s]=vectors.subtract(m[s], vectors.scaler_product(m[s][k],m[k]))
    for k in range(n-1,0,-1):
        if m[k][k]==0:
            continue
        for s in range(k-1,-1,-1):
            i[s]=vectors.subtract(i[s],vectors.scaler_product(m[s][k],i[k]))
    return i
def trace(m):
    t=0
    for i in range(len(m)):
        d+=m[i][i]
    return m
def forbenius_norm(m):
    s=0
    for i in m:
        s+=vectors.squared_magnitude(m[i])
    return s**0.5
def mateix_vector_solver(A,b):
    return matrix_vector_product(matrix_inverse(A),b)
#eigon values are will be deloped in next update
