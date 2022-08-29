#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd


# In[77]:


df=pd.DataFrame({'Loves Popcorn': [1, 1,0,0,1,1,0], 'Loves Soda': [1, 0,1,1,1,0,0],'Age':[7,12,18,35,38,50,83],'Loves Cool As Ice':[0,0,1,1,1,0,0]})


# In[78]:


df


# In[79]:


def calculate_node_discrete(df,column,target):
 tt=0
 tf=0
 ft=0
 ff=0
 for i in range(df.shape[0]):
    if df[column][i] == 1 and df[target][i]==1:
        tt+=1
    elif df[column][i] == 1 and df[target][i]==0:
        tf+=1
    elif df[column][i] == 0 and df[target][i]==1:
        ft+=1
    elif df[column][i] == 0 and df[target][i]==0:
        ff+=1
 return [tt,tf],[ft,ff]


# In[ ]:





# In[80]:


def calculate_gini(node):
    prob=1-(node[0]/(node[0]+node[1]))**2 - (node[1]/(node[0]+node[1]))**2
    return prob


# In[81]:


def calculate_gini_pond_discrete(node):
    prob_true=((node[0][0]+node[0][1])/(node[0][0]+node[0][1]+node[1][0]+node[1][1]))*calculate_gini(node[0])
    prob_false=((node[1][0]+node[1][1])/(node[0][0]+node[0][1]+node[1][0]+node[1][1]))*calculate_gini(node[1])
    return prob_true+prob_false


# In[82]:


def get_means(df,column):
    columna=df[column].sort_values()
    lista=[]
    for i in range(df[column].shape[0]-1):
        lista.append((columna[i]+columna[i+1])/2)
    return lista
        


# In[83]:


def calculate_node_continue(df,columna_promedios,columna_original,target,index):
 tt=0
 tf=0
 ft=0
 ff=0
 for i in range(df[columna_original].shape[0]):
    if df[columna_original][i]<columna_promedios[index]:
        if df[target][i]==1:
            tt+=1
        else:
            tf+=1
    else:
        if df[target][i]==1:
            ft+=1
        else:
            ff+=1
 return [tt,tf],[ft,ff]


# In[84]:


def calculate_gini_pond_continue(df,columna_original,target):
    columna_promedios=get_means(df,columna_original)
    gini_lower=3000
    for i in range(len(columna_promedios)):
        node=calculate_node_continue(df,columna_promedios,columna_original,target,i)
        gini=calculate_gini_pond_discrete(node)
        if gini < gini_lower:
            gini_lower=gini
    return gini_lower


# In[85]:


import operator
def ranking_features(df):
    ranking={}
    for column in df:
        if len(df[column].unique())==2 and (column!=df.columns[-1]):
            node=calculate_node_discrete(df,column,df.columns[-1])
            gini=calculate_gini_pond_discrete(node)
            ranking[column]=gini
        elif len(df[column].unique())>2 and (column!=df.columns[-1]):
            gini=calculate_gini_pond_continue(df,column,df.columns[-1])
            ranking[column]=gini
    
    ranking= sorted(ranking.items(), key=operator.itemgetter(1))
    return ranking


# In[ ]:


def calculate_child_discrete(df,columna_original,columna_nueva,target):
    for i in range(df.shape[0]):
        if df[columna_original][i] == 1 and df[target][i]==1 and df[columna_nueva][i]==1:
            tt+=1
        elif df[columna_original][i] == 1 and df[target][i]==1 and df[columna_nueva][i]==0:
            tf+=1
        elif df[column][i] == 1 and df[target][i]==0:
            tf+=1
        elif df[column][i] == 0 and df[target][i]==1:
            ft+=1
        elif df[column][i] == 0 and df[target][i]==0:
            ff+=1
 return [tt,tf],[ft,ff]


# In[86]:


jeje=ranking_features(df)


# In[87]:


jeje


# In[101]:


get_means(df,"Age")


# In[88]:


print(jeje[0][0])


# In[104]:


def create_tree(df):
    rank=ranking_features(df)
    for i in range(df.shape[1]-1):
        if len(df[rank[i][0]].unique())==2:
            node=calculate_node_discrete(df,rank[i][0],df.columns[-1])
            if calculate_gini(node[0])!=0:
                node=calculate_node_discrete(df,rank[i+1][0],df.columns[-1])
            
        elif len(df[rank[i][0]].unique())>2:
            means=get_means(df,rank[i][0])
            for j in range(len(means)):
                node_2=calculate_node_continue(df,means,rank[i][0],df.columns[-1],j)
      


# In[105]:


create_tree(df)


# In[ ]:


len(df[jeje[3][0]].unique())


# In[ ]:


if len(df[jeje[1][0]].unique())==2:
    node=calculate_node_discrete(df,jeje[1][0],df.columns[-1])
    print(node)
elif len(df[jeje[1][0]].unique())>2:
    node_2=calculate_node_discrete(df,jeje[1][0],df.columns[-1])
    print(node)


# In[ ]:


len(df["Age"].unique())


# In[ ]:


df.iloc[:,-1]


# In[ ]:


(df["Loves Cool As Ice"])!=(df.iloc[:,-1])


# In[110]:


LOL=[1,[12,12,1]],[2,[1,1,1]]


# In[115]:


LOL[1][0]


# In[118]:


LOL[0][1]


# In[117]:


LOL[1][0]

