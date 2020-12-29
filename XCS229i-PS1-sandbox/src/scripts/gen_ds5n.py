import numpy as np
import matplotlib.pyplot as plt

np.random.seed(230)
n = 100
sigma=0.3
factor = 2.0
xmin = -np.pi*factor
xmax = np.pi*factor
x = np.linspace(xmin,xmax,n)
y = np.sin(x)+np.random.normal(0, sigma, n)
plt.scatter(x,y)
plt.savefig('ds5.png')
indices=np.random.permutation(len(x))
train_x=x[indices[0:(int)(0.7*len(x))]]
train_y=y[indices[0:(int)(0.7*len(x))]]
test_x=x[indices[(int)(0.7*len(x)):(int)(0.85*len(x))]]
test_y=y[indices[(int)(0.7*len(x)):(int)(0.85*len(x))]]
val_x=x[indices[(int)(0.85*len(x)):]]
val_y=y[indices[(int)(0.85*len(x)):]]

f= open("train.csv","w+")
f.write('x'+','+'y'+'\n')
for i,j in zip(train_x,train_y):
    f.write(str(i)+','+str(j)+'\n')
f.close()

f= open("test.csv","w+")
f.write('x'+','+'y'+'\n')
for i,j in zip(test_x,test_y):
    f.write(str(i)+','+str(j)+'\n')
f.close()

f= open("valid.csv","w+")
f.write('x'+','+'y'+'\n')
for i,j in zip(val_x,val_y):
    f.write(str(i)+','+str(j)+'\n')
f.close()

f= open("small.csv","w+")
f.write('x'+','+'y'+'\n')
for it,(i,j) in enumerate(zip(train_x,train_y)):
    f.write(str(i)+','+str(j)+'\n')
    if it == 5:
        break
f.close()
