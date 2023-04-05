# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:48:08 2022

@author: Ilord
"""


import networkx as nx
import math
import openpyxl
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import seaborn as sns
from multiprocessing import cpu_count
from multiprocessing import Process,Queue


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def PDs(G,r,ra1):
    
    PDmat=[[1,0],[1+r,0]]
    
    Amat=[[0.5+ra1       ,0.5-1.5*ra1],
          [0.5+0.5*ra1   ,0          ]]
    for i in G.nodes:
        G.nodes[i]["profit"]=0
        G.nodes[i]["act"]=0
    for e in G.edges:
        if G.edges[e]["activate"]==1:
            i1=G.nodes[e[0]]
            i2=G.nodes[e[1]]
            i1["profit"]+=PDmat[i1["strategy"]][i2["strategy"]]
            i2["profit"]+=PDmat[i2["strategy"]][i1["strategy"]]
            i1["act"]+=Amat[i1["strategy"]][i2["strategy"]]
            i2["act"]+=Amat[i2["strategy"]][i1["strategy"]]
    return G

def evo_s(G):
    k=0.1
    
    for i in G.nodes:
        nbrs=list(G[i])
        nbrs_a=[]
        for n in nbrs:
            if G[i][n]["activate"]==1:
                nbrs_a.append(n)
        if len(nbrs_a)==0:
            continue
        comparer = random.choice(nbrs_a)
        M_node = G.nodes[i]["profit"]
        M_comparer = G.nodes[comparer]["profit"]
        
        if (M_node - M_comparer)/k >10:
            continue
        
        P_xy = 1 / (1 + np.exp((M_node - M_comparer)/k))
        if random.random() < P_xy:
            G.nodes[i]["strategy"] = G.nodes[comparer]["strategy"]
    
    return G
        
        
def act_e(G):
    for e in G.edges:
        i1=G.nodes[e[0]]
        i2=G.nodes[e[1]]
        p_a=i1["act"]/len(G[e[0]])+i2["act"]/len(G[e[1]])
        if p_a>=1:
            G.edges[e]["activate"]=1
        elif random.random()<p_a:
            G.edges[e]["activate"]=1
        else:
            G.edges[e]["activate"]=0
    return G

def simu(N,T,ra1,r,Ttest):
    T+=1
    f_coop=[0 for i in range(T)]
    de=[[0 for i in range(15)] for i in range(11)]
    de_point=(T-1)/10
    act_points=[[0 for i in range(21) ] for i in range(int(T/Ttest)+1)]
    c_nbr_a=[[0 for i in range(T)] for i in range(3)]
    d_nbr_a=[[0 for i in range(T)] for i in range(3)]
    
    # set the network
    #G = nx.random_graphs.random_regular_graph(4, N)
    #G=nx.barabasi_albert_graph(N,2)
    G=nx.grid_2d_graph(N,N,periodic=1)
    N=N*N
    #G=nx.random_graphs.watts_strogatz_graph(N, 4, 0.3)
    
    for i in G.nodes:  # initial strategy
        G.nodes[i]["strategy"] = np.random.choice([0, 1])
        
    for e in G.edges:  # initial activation
        G.edges[e]["activate"] = np.random.choice([0,1])
        
    #iteration start################################
    for t in range(T):
        
        
        n_c=N
        for i in G.nodes:
            n_c-=G.nodes[i]["strategy"]
        f_coop[t]=n_c/N
        
        for i in G.nodes:
            
            nbrs=list(G[i])
            nbrs_a=[]
            for n in nbrs:
                if G[i][n]["activate"]==1:
                    nbrs_a.append(n)
            i_act=len(nbrs_a)/len(nbrs)
        
            if (G.nodes[i]["strategy"]):  
                d_nbr_a[0][t]+=i_act
            else:              
                c_nbr_a[0][t]+=i_act
        
        d_nbr_a[0][t]/=(N*(1-f_coop[t]))
        c_nbr_a[0][t]/=(N*(f_coop[t]))
        
        #The relationship between the proportion of cooperative betrayers in neighbors and the proportion of edge activation in the next round
        
        if (t%Ttest==0):
            i_f_c=[]
            for i in G.nodes:                
                sum_c=0
                nbrs=list(G[i])
                for n in nbrs:
                    if G.nodes[n]["strategy"]==0:
                        sum_c+=1
                
                i_f_c.append(sum_c/len(nbrs))
                
        
                    
        PDs(G,r,ra1)
        evo_s(G)
        act_e(G)
        
        if (t%Ttest==0):
            j=0
            f_num_list=[0 for i in range(21)]
            for i in G.nodes:
                nbrs=list(G[i])
                nbrs_a=[]
                for n in nbrs:
                    if G[i][n]["activate"]==1:
                        nbrs_a.append(n)
                i_act=len(nbrs_a)/len(nbrs)
                
                for f in range(21):
                    if ((0.05*f<=i_f_c[j])&(0.05*(f+1)>i_f_c[j])):
                        # if act_points[int(t/Ttest)][f]==-1:
                        #     act_points[int(t/Ttest)][f]=i_act
                        #     f_num_list[f]=1
                        # else:
                            #print(f,end='')
                            act_points[int(t/Ttest)][f]+=i_act
                            f_num_list[f]+=1
                ##############include 1##########################################
                if ((0.05*f<i_f_c[j])&(0.05*(f+1)>=i_f_c[j])):
                        # if act_points[int(t/Ttest)][f]==-1:
                        #     act_points[int(t/Ttest)][f]=i_act
                        #     f_num_list[f]=1
                        # else:
                            #print(f,end='')
                            act_points[int(t/Ttest)][f]+=i_act
                            f_num_list[f]+=1
                
                j+=1
            #print(f_num_list)
            for i in range(21):
                if f_num_list[i]!=0:
                    act_points[int(t/Ttest)][i]/=f_num_list[i]
            
            for i in range(21):
                if f_num_list[i]>5:
                    f_num_list[i]=1
                else: 
                    f_num_list[i]=0
                    act_points[int(t/Ttest)][i]=0
            act_points[int(t/Ttest)].append(f_num_list)
                
    return f_coop,de,c_nbr_a,d_nbr_a,act_points


T=500
N=1500
AVE=10
testT=1


for r0 in range(3):
    r=0.45
    #b=1+r
    fc=[]
    f_ca=[]
    f_da=[]
    f_coop_av_r=[] 
    act_points_av_r=[]
    c_nbr_a_av_r=[]
    d_nbr_a_av_r=[]
    www=[0.2,0.6,3.0]
    ra=www[r0]            #Set parameter combinations
    print(ra)
    for ra0 in range(10):
        f_coop_av=[0 for i in range(T+1)]
        de_av = [[0 for i in range(15)] for i in range(11)]
        act_points_av=[[0 for i in range(21) ] for i in range(int(T/testT)+1)]
        for i in act_points_av:
            i.append([0 for i in range(21)])
        c_nbr_a_av=[[0 for i in range(T+1)] for i in range(3)]
        d_nbr_a_av=[[0 for i in range(T+1)] for i in range(3)]
        N=20+5*ra0       
        #Set parameter combinations，Here we set the network size N
        for ave in range(AVE):
            f_coop_av0,de_av0,c_nbr_a0,d_nbr_a0,act_points0=simu(N, T, ra, r,testT)
            
            for i in range(len(f_coop_av0)):
                f_coop_av[i]+=f_coop_av0[i]/AVE
        
            for i in range(len(c_nbr_a0[0])):
                for j in range(3):
                    c_nbr_a_av[j][i]+=c_nbr_a0[j][i]/AVE
                    d_nbr_a_av[j][i]+=d_nbr_a0[j][i]/AVE
            for i in range(len(act_points_av)):
                for j in range(21):
                    act_points_av[i][j]+=act_points0[i][j]
                for k in range(21):
                    act_points_av[i][-1][k]+=act_points0[i][-1][k]
        
        for i in range(len(act_points_av)):
            for j in range(21):
                if act_points_av[i][-1][j]==0:
                    act_points_av[i][j]=-1
                else:
                    act_points_av[i][j]/=act_points_av[i][-1][j]
            
        #Print evolution results
        print('ra=',ra)
        print("f_coop_av=",f_coop_av)   
        print(N*N,":",f_coop_av)
        print("c_nbr_a_av=",c_nbr_a_av)
        print("d_nbr_a_av=",d_nbr_a_av)
        for i in range(len(act_points_av)):
             print(',',act_points_av[i][0:21])
            
            
      