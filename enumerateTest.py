x = [5,7,8,9,1,2,3]
obj1 = enumerate(x)
print(list(obj1))
idx = max((val, idx) # id of cluster(0,1,2) for x, val is the value of enumerate list
                           for (idx, val) in enumerate(x))
print(idx)
y = max((a, b) for (b, a) in enumerate(x))
print(y)