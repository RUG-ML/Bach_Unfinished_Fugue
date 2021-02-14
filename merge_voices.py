#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


# read in voices
voice_1 = pd.read_csv("F_0.txt", sep="\t", header=None)
voice_2 = pd.read_csv("F_1.txt", sep="\t", header=None)
voice_3 = pd.read_csv("F_2.txt", sep="\t", header=None)
voice_4 = pd.read_csv("F_3.txt", sep="\t", header=None)


# In[9]:


voice_1


# In[7]:


# merge dataframes
df = pd.DataFrame(
    {
        "1": voice_1[0][:550],
        "2": voice_2[0][:550],
        "3": voice_3[0][:550],
        "4": voice_4[0][:550],
    }
)
df


# In[ ]:


# write to new file
f = "F_550.txt"
df.to_csv(f, sep="\t", header=None, index=None)

