import sys
dir_remove = []
for p in sys.path:
    if p.find('python2') !=-1:
        dir_remove.append(p)
for p in dir_remove:
    sys.path.remove(p)

import numpy as np

p = []#np.array([[0,0]])

# for i in range(0,12):
#     for j in range(0,9):
#         p_new = np.array([[50+50*i,50+50*j]])
#         # print (p_new)
#         p = np.concatenate((p,p_new))
        # print (p)
# def unwrap(angle):
#     while angle>=2*np.pi:
#         angle = angle-2*np.pi
#     while angle <2*np.pi:
#         angle = angle+2*np.pi

p_grid = np.array([[0,0]])
for i in range(0,11):
    for j in range(0,11):
        p_new = np.array([[30+45*i,30+45*j]])
        # print (p_new)
        p_grid = np.concatenate((p_grid,p_new))
p_grid = p_grid[1:p_grid.shape[0]+1]
y=[]
i=0

for i, x in enumerate(p_grid):
    if 100 < x[0] < 410 and x[1] < 185:
        y.append(i)# print (np.where(p_grid==x))
        # print np.where
    else:
        continue


y = np.array([y])
# z = np.where(p_grid == y)
