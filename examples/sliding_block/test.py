import numpy as np
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
A = np.ones(5,)
B = np.ones(5,) * 5
print(np.square(B))