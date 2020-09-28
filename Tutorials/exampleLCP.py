import numpy as np
import lemkelcp as lcp 

M = np.array([[2,1], [0,2]])
q = np.array([-1,-2])

sol = lcp.lemkelcp(M,q)

x, code, flag = sol
print('LemkeLCP returned the solution ' + str(x))
print('LemkeLCP returned exit code ' + str(code))
print('LemkeLCP returned exit string ' + flag)