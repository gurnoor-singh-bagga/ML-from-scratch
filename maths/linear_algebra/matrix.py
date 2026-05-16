#matrix opretion
import math.linear_algebra.vectors

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
def LUdecomposition(matrix):
    r,c=shape(matrix)
    if r!=c:
        vectors.error
    n=r
    # create L and U
    L = [[0]*n for _ in range(n)]
    U = [[0]*n for _ in range(n)]

    for i in range(n):

        # Upper triangular
        for j in range(i, n):
            s = 0
            for k in range(i):
                s += L[i][k] * U[k][j]
            U[i][j] = matrix[i][j] - s

        # Lower triangular
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                s = 0
                for k in range(i):
                    s += L[j][k] * U[k][i]
                L[j][i] = (matrix[j][i] - s) / U[i][i]

    return L, U
#the basic lu is written but other functions are not updated nor lu is stoll optimized 
def determinent(m):
    N=shape(m)
    if N[0]!=N[1]:
        vectors.error
    if N==[2,2]:
        return twodeterminent(m)
    else:
        d=1
        l,u=LUdecomposition(m)
        for i in range(N[0]):
            d=d*u[i][i]
        return d
def matrix_inverse(matrix):
    if determinent(matrix)==0:
        vectors.error()
    m=matrix.copy()
    n=len(m)
    l,u=LUdecomposition(m)
    lin=identiy_matrix(n)
    uin=identiy_matrix(n)
    for i in range(n):
        for j in range(i+1,n):
            #lower triangle
            lin[j]=vectors.subtract(lin[j],vectors.scaler_product(l[j][i],lin[i]))
    for i in range(n-1,-1,-1):
        uin[i]=vectors.scaler_product(1/u[i][i],uin[i])
        for j in range(i):
            uin[j]=vectors.subtract(uin[j],vectors.scaler_product(u[j][i],uin[i]))
    return matrix_matrix_product(uin,lin)
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
    ##what will happen if matrix is not sqr ? two cases more eqn less varibles or less eqn more varibles
    #it will be implemented later with eigon values etc....
    return matrix_vector_product(matrix_inverse(A),b)
#eigon values are will be deloped in next update
