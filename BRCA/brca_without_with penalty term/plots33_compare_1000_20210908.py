# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 23:49:30 2020

@author: liugu
"""


import datetime
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt

loss500_ordCOX1 = np.asarray(pd.read_csv("without_BRCA_methylation_mRNA_MT_DSCCA_20210902.csv",header=None))  
loss500_M_m = np.asarray(pd.read_csv("all_with_penalty_BRCA_methylation_mRNA_MT_DSCCA_20210902.csv",header=None))
#loss500_auxiliary= np.asarray(pd.read_csv("hist.history['loss1000_ml_ordCOX20201220']_without_penalty_term_only_auxiliary_loss.csv",header=None))
#loss500_mRNA= np.asarray(pd.read_csv("hist.history['loss500_mRNA'].csv",header=None))
loss500_ordCOX2=[]
#for i in range(2000):
#    if i>500:
#        loss500_ordCOX11=loss500_ordCOX1[500]-i/100+(loss500_ordCOX1[i]-loss500_ordCOX1[i-1])
#        loss500_ordCOX2.extend(loss500_ordCOX11)
#    if i<=500:
#        loss500_ordCOX2.extend(loss500_ordCOX1[i])
#
#for i in range(480,520):
#    loss500_ordCOX2[i]=loss500_ordCOX1[i]+i/100
epoch=np.arange(1000)
#plt.gca().set_color_cycle(['red','green','blue','black'])


plt.plot(epoch,loss500_ordCOX1[0:1000],color='red')
plt.plot(epoch,loss500_M_m[0:1000],color='blue' )
#plt.plot(epoch,loss500_auxiliary[0:1000],color='red' )
#plt.plot(epoch,loss500_Methylation )
#plt.plot(epoch,loss500_mRNA )
plt.title('BRCA Loss Curves (With/Without Penalty Term)')
plt.ylabel('training-loss')
plt.xlabel('epoch')
plt.legend(['The method without penalty term','The method with penalty term'], loc='upper right')
plt.savefig('brca_Loss Curves2_with_without penalty term.jpg', dpi=300) #指定分辨率保存
plt.show()
#plt.savefig('Loss Curves.jpg', dpi=300) #指定分辨率保存
#a/${m_2}$ 
