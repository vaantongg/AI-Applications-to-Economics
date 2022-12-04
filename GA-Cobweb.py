#!/usr/bin/env python
# coding: utf-8

# ### GA Project
# ### Van Tong
# <hr style="border:2px solid gray"> </hr>
# 
# # Outline
# ### Part 0. 
# ### Part 1. 
# ### Part 2. 
# ### Part 3. 
# <hr style="border:2px solid gray"> </hr>
# 
# 

# # Part 0
# ### Question 12
# ##### (a) purpose of the mutation operator, what would happen if it was removed from the GA?
# The mutation operator give strings that never been tried enter the population. Without the mutation operator, strings what were never been tried will have no chance of being introduced into the population, including the "good" strings that yield optimal outcome
# ##### (b) purpose of the crossover operator, what would happen if it was removed from the GA?
# The crossover operator allows for better firms (since it's after reproduction) to exchange information and ideas. Without this operator, convergence may be slower
# ##### (c)  purpose of determining fitness, what would happen if it was removed from the GA? 
# Fitness is the key criteria that determine the likelihood for a strings to be copied to the next generation's population. Without a well defined fitness, decision will be purely stochastic.
# ##### (d) proportional selection, what would happen if instead each string had an equal chance of being selected?
# The purpose of proportional selection is to choose better performing strings with higher chance into the next generation's population, this makes sure the population is updated in a way that next population is relatively performing better than the previous. Proportional selection works together with fitness, without proportional selection. decision will be purely stochastic.
# <hr style="border:2px solid gray"> </hr>
# 

# # Part 1: Define simulation function
# 
# ### Note: given this parameter set, the Rational Expectation Equilibirum Quantity is ~4.55
# ### Note: under basic GA, if setting q_max = 8, the system fluctuates without settling at the REE (perfect foresight competitive equilibrium)
# ### Note: under basic GA, if setting q_max = 4, the system fluctuates and seems to converging to ~ 2.5, seems like this is the colusive equilibrium
# 
# Note: the results below is under q_max = 8

# In[1]:


get_ipython().run_line_magic('reset', '-f')

import random
from scipy.stats import truncnorm
import numpy as np
from sys import exit
import plotly
import plotly.graph_objs as go
from plotly import subplots
plotly.offline.init_notebook_mode(connected=True)
import matplotlib.pyplot as plt


# PARAMETERS DESCRIPTION
rounds = 150
runs = 20
I = 20 #number of firms
J = 20 #size of GA popl

##production technology parameters x and y
x = 0
y = 0.016

##experimentation parameters
pcross = 0.9
pmut = 0.033
K = int('1'*J,2)/8 #so that q_max is 8, for a length 20 string

##demand schedule
A = 2.84
B = 0.0152

## rational expectation equilibrium
RE_price = (A*y + B*x)/(y + B)
RE_quantity = (A - x)/((B + y)*I)


## QUESTION 1
def randominitialize(I, J):
    pop=[]
    for i in range(I):
        string=[]       
        for j in range(J):
            string.append(random.choice([0,1])) 
        pop.append(string)  
        
    return pop



## QUESTION 2
def decoding(I,pop):
    pop_x = []
    for i in range(I):
        summ = []
        for j in range(J):
            temp = pop[i][j]**(j+1)*2**(j+1-1)
            summ.append(temp)
        x = sum(summ)  #decoded string for each i
        pop_x.append(x)
    return pop_x

def normalize(I,pop_x,K):
    pop_q = []
    for i in range(I):
        pop_q.append(pop_x[i]/K)
    return pop_q




## QUESTION 3
def profit(pop_q, price):
    payoff = []
    for i in range(I):
        payoff.append(price*pop_q[i]-x*pop_q[i]-0.5*y*I*(pop_q[i])**2)
    return payoff

def fitness(payoff):
    fit = []
    for i in range(I):
        temp = payoff[i]/sum(payoff)
        if temp >= 0:
            fit.append(temp)
        if temp < 0:
            fit.append(0.00001)
    return fit



## QUESTION 7
def reproduction(pop,fitness):
    temp_pop = []
    for i in range(I):
        x = random.uniform(0,1)
        added_prob = 0
        for item, item_prob in zip(pop, fitness):
            added_prob += item_prob
            if x < added_prob:
                select = item
                break
        temp_pop.append(select)
    return temp_pop

## QUESTION 4 and QUESTION 5
def crossover(temp_pop, pcross): # unique couples
    offsprings = []
    temp_pop_index = list(range(0,I))
    for i in range(int(I/2)):
        parents_index = random.sample(temp_pop_index,2)
        parent1 = temp_pop[parents_index[0]]
        parent2 = temp_pop[parents_index[1]]
        temp_pop_index.remove(parents_index[0])    
        temp_pop_index.remove(parents_index[1])   #removes this couples out of temp_pop, so no repeated couple
        
        x = random.uniform(0,1)
        k = random.randint(0,J-1)
        
        if x <= pcross:
            offspring1 = parent1[0:k] + parent2[k:J]
            offspring2 = parent2[0:k] + parent1[k:J]
        else:
            offspring1 = parent1
            offspring2 = parent2
        offsprings.append(offspring1)
        offsprings.append(offspring2)
    return offsprings



## QUESTION 6
def mutation(temp_pop, pmut):
    new_pop = [[0]*J]*I
    for i in range(I):
        for j in range(J):
            x = random.uniform(0,1)
            if x <= pmut:
                new_pop[i][j] = 1 - temp_pop[i][j]
            else:
                new_pop[i][j] = temp_pop[i][j]
    return new_pop



########################################## SIMULATION FUNCTION ##################################################    

## QUESTION 8
prices_t_of_all_runs = [[] for i in range(rounds)] #each element stores all prices of time t of all runs
A_runs_prices = [[] for i in range(rounds)]
A_runs_quantities = [[] for i in range(rounds)]
quantities_t_of_all_runs = [[] for i in range(rounds)]

random.seed(7)
def simulate(A, B, x, y, K, pcross, pmut, I, J, rounds, runs):
    for sims in range(runs):
        #print("run", sims)
        prices = []
        ave_quantities = []

        pop = randominitialize(I, J)
        for t in range(rounds):
            pop_x = decoding(I,pop)
            pop_q = normalize(I,pop_x,K)                  # normalize all strings to firms' quantity decisions

            price = A - B*sum(pop_q)                      # market price realized
            pay = profit(pop_q, price)
            fit = fitness(pay)                            # firms' profit and fitness

            temp_pop = reproduction(pop,fit)              # population updating starts
            offsprings = crossover(temp_pop, pcross)
            new_pop = mutation(offsprings, pmut)

            pop = new_pop                                # new pop

            prices.append(price)
            ave_quantities.append(np.mean(pop_q))

            prices_t_of_all_runs[t].append(price)
            quantities_t_of_all_runs[t].append(np.mean(pop_q))

    for t in range(rounds):
        A_runs_prices[t] = np.mean(prices_t_of_all_runs[t])
        A_runs_quantities[t] = np.mean(quantities_t_of_all_runs[t])
    return A_runs_prices, A_runs_prices
########################################## SIMULATION FUNCTION END ##############################################    




# In[2]:


randominitialize(I,J)


# <hr style="border:2px solid gray"> </hr>

# # Part 2: simulation with pmut = 0.033, graphs is below

# In[3]:



########################################## pmut = 0.033 #########################################################
simulate(A, B, x, y, K, pcross, pmut, I, J, rounds, runs)
########################################## GRAPH ################################################################    
## GRAPH 1 ##
trace1 = go.Scatter(
    x = list(range(rounds)), #x-axis
    y = list(A_runs_prices), #y-axis
    mode = 'lines+markers',
    name = 'lines'
)                
layout1 = go.Layout(
    title=('Single population baisc GA pattern of price (' + str(I) + ' firms, J=' + str(J) + ', runs=' + str(runs) +', pmut=' + str(pmut)+')'),
    width=800,
    height=600,
    xaxis=dict(
        title='round',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='prices',
        nticks=10,
        range=[1,2.5],
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
data1 = [trace1]
figure1 = go.Figure(data=data1,layout=layout1)
figure1.add_shape(type='line',
                x0=0,
                y0=RE_price,
                x1=rounds,
                y1=RE_price,
                line=dict(color='Red',),
                xref='x',
                yref='y'
)
figure1.add_annotation(x=rounds, y=RE_price,
            text="RE price",
            showarrow=True,
            arrowhead=1)
figure1.show()         




## GRAPH 2 ##
trace2 = go.Scatter(
    x = list(range(rounds)), #x-axis
    y = list(A_runs_quantities), #y-axis
    mode = 'lines+markers',
    name = 'lines'
)                
layout2 = go.Layout(
    title=('Single population baisc GA pattern of quantity (' + str(I) + ' firms, J=' + str(J) + ', runs=' + str(runs) +', pmut=' + str(pmut)+')'),
    width=800,
    height=600,
    xaxis=dict(
        title='round',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='quantity',
        nticks=10,
        range=[0,8],
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
data2 = [trace2]
figure2 = go.Figure(data=data2,layout=layout2)

figure2.add_shape(type='line',
                x0=0,
                y0=RE_quantity,
                x1=rounds,
                y1=RE_quantity,
                line=dict(color='Red',),
                xref='x',
                yref='y'
)
figure2.add_annotation(x=rounds, y=RE_quantity,
            text="RE quantity",
            showarrow=True,
            arrowhead=1)
figure2.show() 


# <hr style="border:2px solid gray"> </hr>
# 

# # Part 3: simulation with pmut = 0.0033, graphs is below

# In[3]:


simulate(A, B, x, y, K, pcross, 0.0033, I, J, rounds, runs)

## GRAPH 3 ##
trace3 = go.Scatter(
    x = list(range(rounds)), #x-axis
    y = list(A_runs_prices), #y-axis
    mode = 'lines+markers',
    name = 'lines'
)                
layout3 = go.Layout(
    title=('Single population baisc GA pattern of price (' + str(I) + ' firms, J=' + str(J) + ', runs=' + str(runs) +', pmut=' + str(pmut)+')'),
    width=800,
    height=600,
    xaxis=dict(
        title='round',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='prices',
        nticks=10,
        range=[1,2.5],
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
data3 = [trace3]
figure3 = go.Figure(data=data3,layout=layout3)
figure3.add_shape(type='line',
                x0=0,
                y0=RE_price,
                x1=rounds,
                y1=RE_price,
                line=dict(color='Red',),
                xref='x',
                yref='y'
)
figure3.add_annotation(x=rounds, y=RE_price,
            text="RE price",
            showarrow=True,
            arrowhead=1)
figure3.show()         




## GRAPH 4 ##
trace4 = go.Scatter(
    x = list(range(rounds)), #x-axis
    y = list(A_runs_quantities), #y-axis
    mode = 'lines+markers',
    name = 'lines'
)                
layout4 = go.Layout(
    title=('Single population baisc GA pattern of quantity (' + str(I) + ' firms, J=' + str(J) + ', runs=' + str(runs) +', pmut=' + str(pmut)+')'),
    width=800,
    height=600,
    xaxis=dict(
        title='round',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='quantity',
        nticks=10,
        range=[0,8],
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
data4 = [trace4]
figure4 = go.Figure(data=data4,layout=layout4)

figure4.add_shape(type='line',
                x0=0,
                y0=RE_quantity,
                x1=rounds,
                y1=RE_quantity,
                line=dict(color='Red',),
                xref='x',
                yref='y'
)
figure4.add_annotation(x=rounds, y=RE_quantity,
            text="RE quantity",
            showarrow=True,
            arrowhead=1)
figure4.show() 


# In[14]:


import matplotlib.pyplot as plt


# In[5]:


randominitialize(I, J)


# In[ ]:




