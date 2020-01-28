#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:46:01 2020

@author: rutvik
"""
import tensorflow as tf



class network(tf.keras.Model):
  def __init__(self, hidden_dim, action_space, reg = None):
    super(network,self).__init__()

    self.l1 = tf.keras.layers.Dense(hidden_dim,activation='relu',kernel_regularizer=reg, bias_regularizer=reg)
    
    self.l2 = tf.keras.layers.Dense(hidden_dim,activation = 'relu',kernel_regularizer=reg, bias_regularizer=reg)

    self.out = tf.keras.layers.Dense(action_space,kernel_regularizer=reg, bias_regularizer=reg,)
    
  def call(self,obs):
    x = self.l1(obs)
    x = self.l2(x)
    return self.out(x)