import numpy as np
from scipy.stats import norm
class ParentClass: 
    def __init__(self):
        self.f()
        self.h()
    def f(self): 
        print("Hi!"); 
    def h(self):
        print("bye")

class ChildClass(ParentClass): 
    def __init__(self):
        super(ChildClass, self).__init__()
        # print(np.zeros(5,))
        
        # self.f()
        # self.h()
    def f(self):
        
        # print(super(ChildClass, self))
        # # print(super())
        # self.h()
        print("Hello!")
    # def h(self):
    #     super(ChildClass, self).h()
    #     print("bye bye")

# A = ChildClass()
# A.f()
# A.h()
A = np.array([-2, -1, 0, 1, 2])
B = np.ones(5,) * 5
mu = 2
sigma  = 1
norm_A = norm.pdf(A, mu, sigma)
print(A)
print(np.sum(A, axis = 0))