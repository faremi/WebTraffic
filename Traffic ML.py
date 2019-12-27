#!/usr/bin/env python
# coding: utf-8

# In[4]:


from os import *
import scipy as nd
import matplotlib.pyplot as plt


# In[94]:


data = nd.genfromtxt("/home/wizard/BuildingMachineLearningSystemsWithPython-master/ch01/data/web_traffic.tsv", delimiter="\t")
colors = ['g', 'k', 'b', 'g', 'r', 'm']
linestyles = ['-', '-.', '--', ':', '-', '-']
x = data[:, 0]
y = data[:, 1]
x = x[~nd.isnan(y)]
y = y[~nd.isnan(y)]


# In[80]:


def model_plot(x, y, models, mx=None, ymax=None, xmin=None):
    plt.figure(num = None, figsize = (10,5))
    plt.clf()
    plt.scatter(x,y, s =10)
    plt.title("Machine Learning for Web Traffic")
    plt.xlabel("Weeks")
    plt.ylabel("Traffic")
    plt.xticks([w * 7 * 24 for w in range(10)], ['Week %i' %w for w in range(10)])
    plt.autoscale(tight = True)
    plt.ylim(ymin =0)
    
    if models:
        if mx is None:
            mx = nd.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, linestyles, colors):
            # print "Model:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")
    
    if ymax:
        plt.ylim(ymax = ymax)
    if xmin:
        plt.xlim(xmin = xmin)
    plt.grid()
    plt.show()
    
    


# In[76]:


f1 = nd.poly1d(nd.polyfit(x,y,1))
f2 = nd.poly1d(nd.polyfit(x,y,2))
f3 = nd.poly1d(nd.polyfit(x,y,3))
f4 = nd.poly1d(nd.polyfit(x,y,4))
f10 = nd.poly1d(nd.polyfit(x,y,10))
f20 = nd.poly1d(nd.polyfit(x,y,20))


# In[104]:


infletion = 3 * 7 * 24
xa = x[:infletion]
ya = y[:infletion]
xb= x[infletion:]
yb = y[infletion:]
model_plot(xa,ya, None)
model_plot(xb, yb, None)


# In[105]:


model_plot(x,y,[f1,f2,f3,f4,f10,f20])
model_plot(xa,ya,[f1,f2,f3,f4,f10,f20])
model_plot(xb,yb,[f1,f2,f3,f4,f10,f20])


# In[107]:


def error(m,x,y):
    return nd.sum((m(x)-y)**2)


# In[194]:


for f in [f1, f2, f3, f4, f10, f20]:
    print("Error d=%i: %f" % (f.order, error(f, x, y)))


# In[193]:


for f in [f1, f2, f3, f4, f10, f20]:
    print("Error d=%i: %f" % (f.order, error(f, xa, ya)))


# In[192]:


for f in [f1, f2, f3, f4, f10, f20]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))


# In[216]:


fraction = 0.2
value = int(fraction * len(xa))
shuffled = nd.random.permutation(list(range(len(xa))))
test = sorted(shuffled[:value])
train = sorted(shuffled[value:])
ft1 = nd.poly1d(nd.polyfit(xa[train], ya[train], 1))
ft2 = nd.poly1d(nd.polyfit(xa[train], ya[train], 2))
ft3 = nd.poly1d(nd.polyfit(xa[train], ya[train], 3))
ft4 = nd.poly1d(nd.polyfit(xa[train], ya[train], 4))
ft5 = nd.poly1d(nd.polyfit(xa[train], ya[train], 10))
ft6 = nd.poly1d(nd.polyfit(xa[train], ya[train], 20))


# In[218]:


model_plot(x,y,[ft1,ft2,ft3,ft4,ft5,ft6])


# In[227]:


from scipy.optimize import fsolve
reached_max = fsolve(ft4 - 2000, x0=800) / (7 * 24)
print("100,000 hits/hour expected at week %f" % reached_max[0])


# In[220]:





# In[ ]:




