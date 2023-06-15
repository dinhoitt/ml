# -*- coding: utf-8 -*-
"""
Life is what you make of it 

Written by Dinho_itt (this is my instagram id)
"""

import math as m
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

plt.close("all")
plt.xlabel("x1") #x1~x10까지 변경 가능
plt.ylabel("y")
dfLoad =  pd.read_csv("file:///C:/Users/Home/Desktop/training.txt", sep ="\s+")
xxRaw = dfLoad["x1"] #x1~x10까지 변경 가능
yyRaw = dfLoad["y"]
yyRawNP = np.array(yyRaw)
plt.plot(xxRaw, yyRaw, "r.")

import math as m
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

plt.close("all")
plt.title("(x1,y) plot")
plt.xlabel("x1")
plt.ylabel("y")

dataset = pd.read_csv("file:///C:/Users/Home/Desktop/training.txt",
                     sep = "\s+")
xraw = np.array(dataset.values[:,0])
yraw = np.array(dataset.values[:,1])



def gradient_descent(x,y):
    m_curr = -0.5
    b_curr = -0.5
    iterations = 1000
    n = len(x)
    learning_rate = 0.01

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        plt.plot(m_curr,b_curr,marker='o',
                 color='blue',markerfacecolor='red',
                 markersize=10,linestyle='dashed')

x=np.array(dataset['x1'])
y=np.array(dataset['y'])

gradient_descent(x,y)
