a = 2
x = [(1,2,3,4),(4,5,6,7),(8,9,10)]

print(list(zip(x)))
print(list(zip(*x)))
sumx = map(sum, list(zip(*x)))
print(sumx)
center = [x/a for x in sumx]
print(center)
print(list(sumx))
if (8.06880934e-03 > 1): 
    print("8.06880934e-03 < 1")