#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from scipy.stats import truncnorm
import numpy as np
from sys import exit
import plotly
import plotly.graph_objs as go


##THE PARAMETERS DESCRIPTION (example: unstable case, set 7 in the paper)
############################################################################################################################

##the environment parameters
rounds=10 #number of rounds per run (each agent make decision and learning for 10 rounds)
runs=10
I=10 #number of firms in a simulation
J=30 #the number of alternatives in the remembered strategy set

##action space [lower,upper]
S_upper=10
S_lower=0

##production technology parameters x and y
x=0
y=0.016

##experimentation parameters
pmut=0.033 #rate of mutation of value
sigmav=1

##demand schedule
A=2.296
B=0.0168
############################################################################################################################





## IEL PROCEDURES
############################################################################################################################
#initialization of remembered strategy set
def randominitialize(I, J, S_upper, S_lower):
    W=[] #initial utility
    St=[] #initial remembered strategy set 
    
    for i in range(I): #for each firm
        temp=[2]*len(range(J)) #as J is the length of the set
        W.append(temp)
        
        
    for i in range(I):
        Sit=[]
        
        for j in range(J):
            Sit.append(random.uniform(S_lower,S_upper)) # Sit contains J elements draw form the strategy set [S_lower,S_upper]
        St.append(Sit)  #St contains I elements, each belongs to a firm
        
    return St, W  #W=[[Wi],..,[WI]], similar for St

#strategy selection in a strategy set
def selectionfori(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities): #this zip() return a pair of tuples where 1st element in each are paired and so on
        cumulative_probability += item_probability
        if x < cumulative_probability: break
    return item

#ASSIGN PROBABILITY CORRESPONDING TO PROFIT
def choiceprobabilitiesfori(profits):
    choicepiti=[]
    e=min(0,min(profits))
    if e <= 0:
        #print("e<0")
        for j in range(J):
            profits[j] -= e
            #print(profits[j])
    sumw=sum(profits)
    if sumw == 0:
        exit("error: sumw=0")
    for j in range(J):
        choicepiti.append(profits[j]/float(sumw))
    return choicepiti

##EXPERIMENTATION
def Vexperimentationfori(strategyset):
    for j in range(J):
        if random.uniform(0, 1) < pmut:
            centers = strategyset[j]
            r = (truncnorm.rvs((S_lower - centers) / float(sigmav),  #note: truncnorm.rvs(a,b,size=100)
                                (S_upper - centers) / float(sigmav),
                                loc=centers, scale=sigmav, size=1))
            strategyset[j] = np.array(r).tolist()[0]
    return strategyset

##CALCULATING FOReGONE PROFIT
def foregoneprofit(strategy, past_actions, player_name):
    temp_actions=list(past_actions) #last period action of each firm
    temp_actions[player_name]=strategy
    profit=[]
    for i in range(I):
        P=A-B*sum(temp_actions)
        #print(P)
        profit.append((A-B*sum(temp_actions)*temp_actions[i]-x*temp_actions[i]-0.5*y*I*(temp_actions[i])**2))
    utility=profit[i]
    return utility



def updateWfori(Set,past_actions,player_name): #update utility set
    W=[]

    for j in range(J):
        W.append(foregoneprofit(Set[j],past_actions,player_name))
        #W.append(10-Set[j])
    return W



def replicatefori(strategyset, utilities):
    newS=[0]*J
    newW=[0]*J
    for j in range(J):
        j1=random.randrange(J)
        j2=random.randrange(J)
        newS[j]=strategyset[j2]
        newW[j]=utilities[j2]
        if utilities[j1]>utilities[j2]:
            newS[j]=strategyset[j1]
            newW[j]=utilities[j1]
    return newS, newW
############################################################################################################################




## VARIABLES OF INTERESTED
############################################################################################################################
a_mean_round_run = [[] for i in range(rounds)]  # stores the mean quantity chosen by each round each run
# [[round1run1,round1run2,....][round2run1,round2run2...]....]
P_round_run = [[] for i in range(rounds)]
a_mean_round= [0]*rounds # stores the mean contribution  of all rounds by each round
P_round =[0]*rounds
############################################################################################################################




##SIMULATION FUNCTION
############################################################################################################################
def simulation(rounds, runs, I, J, S_upper, S_lower, x, y, pmut, sigmav, A, B):
    for sims in range(runs):
        random.seed()
        S = []  # stores strategy sets for each run
        a = []  # stores all actions for a run
    
    
        currentstrat = [0] * I
    
        [St, W] = randominitialize(I, J, S_upper, S_lower)
    
        for t in range(rounds):
            at = []  # initializes actions for round t
        
            if t==0:
                for i in range(I):
                    p = choiceprobabilitiesfori(W[i])
                    at.append(selectionfori(St[i], p))
                P_round_run[t].append(A-B*sum(at))   
                a_mean_round_run[t].append(np.mean(at))
                S.append(St)
                a.append(at)
            else:
                for i in range(I):
                    St[i] = Vexperimentationfori(St[i])

                    W[i] = updateWfori(St[i], a[len(a)-1], i) #(strategy, past_actions, player_name)
                    (St[i], W[i]) = replicatefori(St[i], W[i])
                for i in range(I):
                    p = choiceprobabilitiesfori(W[i])
                    at.append(selectionfori(St[i], p))
                
                S.append(St)
                a.append(at)
                P_round_run[t].append(A-B*sum(at))
                a_mean_round_run[t].append(np.mean(at)) #a_mean_round_run stores all average action of each round

        print("running", sims)


    for t in range(rounds):
        a_mean_round[t]=np.mean(a_mean_round_run[t]) #average action of round t in all run
        P_round[t]=np.mean(P_round_run[t]) #average price of round t in all run
    return a_mean_round, P_round
############################################################################################################################

#the cobweb unstable case, set 7 in the paper
simulation(rounds, runs, I, J, S_upper, S_lower, x, y, pmut, sigmav, A, B)

#create traces
trace0 = go.Scatter(
    x = list(range(rounds)), #x-axis
    y = list(a_mean_round), #y-axis
    mode = 'lines+markers',
    name = 'lines'
)
layout = go.Layout(
    title='fill title later',
    width=800,
    height=600,
    xaxis=dict(
        title='period',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='quantity produce',
        nticks=10,
        range=[0,10],
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
data = [trace0]
figure = go.Figure(data=data,layout=layout)
plotly.offline.plot(figure, filename='quantity.html')
print("average quantity produced per ground",a_mean_round)



#create trace for Price
trace1 = go.Scatter(
    x = list(range(rounds)), #x-axis
    y = list(P_round), #y-axis
    mode = 'lines+markers',
    name = 'lines'
)
layout2 = go.Layout(
    title='fill title later',
    width=800,
    height=600,
    xaxis=dict(
        title='period',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='price',
        nticks=10,
        range=[0,10],
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
data2 = [trace1]
figure = go.Figure(data=data2,layout=layout2)
plotly.offline.plot(figure, filename='price.html')
print("average price per ground of all runs",P_round)









# In[ ]:




