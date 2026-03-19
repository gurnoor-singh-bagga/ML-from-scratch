#after thinking for a time , we will be using inbuilt math library 
# for computional ease for function like log and exp and wont be used anywhere else
# these function can be build but they will be build only for learning purpose with tylor series 

# the functions made for learning purpose like exp and log which need to be optimize properly
#  before use which are  out of scope for us and diffrent kind of project itself

import math

def sigmoid(x):
    if (x<0):
        return math.exp(x)/(1+math.exp(x))
    else:
        return 1/(1+math.exp(-x))
