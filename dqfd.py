#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 23:14:18 2020

@author: rutvik
"""

import tensorflow as tf
import gym
import numpy as np
from collections import deque
import datetime as dt

from models import network


from exp_replay import PartitionedMemory
from utils import load_demo_data_from_file, reward_threshold_subset, RocketProcessor
processor = RocketProcessor()


class agent:

  def __init__(self, state_space, action_space, trajectory_len, hidden_dim, learning_rate, GAMMA, memory, lambda_list = [1,1,1,0.00001]):
      
    self.state_space = state_space
    self.action_space = action_space
    self.learning_rate = learning_rate
    self.memory = memory
    self.trajectory_len = trajectory_len
    # self.replay_buffer = replay_buffer()

    # self.TAU = TAU
    self.GAMMA = GAMMA
    
    self.q_net = network(hidden_dim,action_space,reg = tf.keras.regularizers.l2(lambda_list[3]))
    self.target_q_net = network(hidden_dim,action_space, None)
    self.update_target()
    self.lambda_list = lambda_list

    self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    self.time_step = 0
    self.UPDATE_TARGET_NET = 1000
      
  def update_target(self):
    self.target_q_net.set_weights(self.q_net.get_weights())
  
  def choose_action(self,obs):
      act = self.q_net(obs)
      return act
  
  
  def compute_dqn_loss(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch):
    q_vals = self.q_net(tf.convert_to_tensor(next_state_batch,dtype = tf.float32)).numpy()
    target_q_vals = self.target_q_net(tf.convert_to_tensor(next_state_batch,dtype = tf.float32)).numpy()
    
    fin_target = self.q_net(tf.convert_to_tensor(state_batch,dtype = tf.float32)).numpy()
    # print("HERE:",len(state_batch))
    for b in range(len(state_batch)):
      
      a_val = tf.argmax(q_vals[b])
      if not done_batch[b]:
        # print(b,a_val)
        
        fin_target[b][action_batch[b]] = reward_batch[b] + self.GAMMA*target_q_vals[b][a_val]
        # print(action_batch[b])
      else:
        fin_target[b][action_batch[b]] = reward_batch[b]
    
    orig_q_vals = self.q_net(tf.convert_to_tensor(state_batch,dtype = tf.float32))
    return tf.reduce_sum(tf.square(fin_target-orig_q_vals), axis = 1), tf.reduce_sum(tf.abs(fin_target - orig_q_vals), axis=1)
  

  def compute_n_step_dqn_loss(self,state_batch,action_batch,n_step_state_batch,n_step_reward_batch,n_step_done_batch,n_count):   
    q_vals = self.q_net(tf.convert_to_tensor(n_step_state_batch,dtype = tf.float32)).numpy()
    target_q_vals = self.target_q_net(tf.convert_to_tensor(n_step_state_batch,dtype = tf.float32)).numpy()
    
    fin_target = self.q_net(tf.convert_to_tensor(state_batch,dtype = tf.float32)).numpy()
    
    for b in range(len(state_batch)):
      a_val = tf.argmax(q_vals[b])
      if not n_step_done_batch[b]:
        fin_target[b][action_batch[b]] = n_step_reward_batch[b] + self.GAMMA**n_count[b]*target_q_vals[b][a_val]
      else:
        fin_target[b][action_batch[b]] = n_step_reward_batch[b]

    return tf.reduce_sum(tf.square(fin_target- self.q_net(tf.convert_to_tensor(state_batch,dtype = tf.float32))),axis = 1)

  def loss_l(self,ae,a):
    if ae == a:
      return 0.8  
    else:
      return 0

  def compute_je_loss(self,state_batch, action_batch, is_demo_batch):
    q_val = self.q_net(tf.convert_to_tensor(state_batch,dtype = tf.float32))
    action_batch_r = tf.reshape(action_batch,[len(state_batch),1])
    exp_val = tf.gather_nd(q_val,action_batch_r, batch_dims = 1)
    
    exp_margin = tf.one_hot(action_batch, self.action_space, on_value = 0, off_value = 1)

    exp_margin = 0.8*tf.cast(exp_margin,tf.float32)
    # q_val = [32,4] 
    add_res = tf.add(q_val,exp_margin)
    max_margin = tf.reduce_max(add_res, axis = 1) 
    
    max_margin_2 = tf.subtract(max_margin, exp_val)
    jeq = tf.multiply(is_demo_batch,max_margin_2)

    return jeq
  
  
  def get_ind_batches(self,minibatch):      
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch = []
    is_demo_batch = []
    n_step_reward_batch = []
    n_step_state_batch = []
    n_step_done_batch = []
    n_count = []
    for i in range(len(minibatch)):
      state_batch.append(minibatch[i][0][0])
      action_batch.append(minibatch[i][1])
      reward_batch.append(minibatch[i][2])
      next_state_batch.append(minibatch[i][3][0])
      done_batch.append(minibatch[i][4])
      is_demo_batch.append(minibatch[i][5])
      n_step_reward_batch.append(minibatch[i][6])
      n_step_state_batch.append(minibatch[i][7][0])
      n_step_done_batch.append(minibatch[i][8])
      n_count.append(minibatch[i][9])
   
    return state_batch, action_batch, reward_batch, next_state_batch, done_batch, is_demo_batch, n_step_reward_batch, n_step_state_batch, n_step_done_batch, n_count

  def compute_losses(self, minibatch, isWeights):
    
    state_batch, action_batch, reward_batch, next_state_batch, done_batch, is_demo_batch, n_step_reward_batch, n_step_state_batch, n_step_done_batch, n_count = self.get_ind_batches(minibatch)

    dqn_loss, abs_err = self.compute_dqn_loss(state_batch,action_batch,reward_batch,next_state_batch,done_batch)
    dqn_loss = tf.reduce_mean(isWeights*dqn_loss)
    n_step_dqn_loss = tf.reduce_mean(isWeights*self.compute_n_step_dqn_loss(state_batch,action_batch,n_step_state_batch,n_step_reward_batch,n_step_done_batch,n_count))
    je_loss = tf.reduce_mean(self.compute_je_loss(state_batch,action_batch,is_demo_batch))
    l2_loss = tf.reduce_mean(tf.add_n(self.q_net.losses))
    tot_loss = self.lambda_list[0]*dqn_loss + self.lambda_list[1]*n_step_dqn_loss + self.lambda_list[2]*je_loss + l2_loss

    return tot_loss, abs_err
          
          
  def train_step(self, mini_batch, idxs, isWeights, update = True):
    self.time_step+=1
    with tf.GradientTape() as tape:
      loss, abs_err = self.compute_losses(mini_batch,isWeights)   

    grads =  tape.gradient(loss,self.q_net.trainable_variables)
    self.optimizer.apply_gradients(zip(grads,self.q_net.trainable_variables))  

    if update and self.time_step % self.UPDATE_TARGET_NET:
      self.update_target()
    
    self.memory.update_priorities(idxs, abs_err.numpy())
    return loss

  def train(self, steps, BATCH_SIZE):
    idxs, mini_batch, isWeights = self.memory.sample(steps, BATCH_SIZE, self.trajectory_len, self.GAMMA)
    l = self.train_step(mini_batch, idxs, isWeights, update = False)	
    return l
  
  def pretrain(self,pretrain_steps, BATCH_SIZE):
    for i in range(pretrain_steps):
      
      idxs, mini_batch, isWeights = self.memory.sample(i, BATCH_SIZE, self.trajectory_len, self.GAMMA)
      l = self.train_step(mini_batch, idxs, isWeights, update = True)
      
      if i%10 == 0 : 
        print("Step {}\tloss : {}".format(i,l))
    self.time_step = 0

  def add_transition(self,s,a,r,t,is_exp):
    self.memory.append(s, a, r, t, is_exp)

  def save_weights(self,checkpoint_path):
    self.q_net.save_weights(checkpoint_path)
    
  def load_weights(self,checkpoint_path):
    self.q_net.load_weights(checkpoint_path)
    self.update_target()



def test(dqfd_agent,env,state_space,action_space,NUM_EPISODES):
  for ep in range(NUM_EPISODES):
      
    state = env.reset()
    score = 0
    ep_steps = 0
    while(True):
      env.render()
      action = dqfd_agent.choose_action(tf.reshape(state,[1,state_space]))
      action = np.argmax(action.numpy())
      next_state,reward,done,_ = env.step(action)
      score += reward
      state = next_state		
      if done: 
        print('Episode {}, reward : {}'.format(ep,score))
        break
  
def run_dqfd(env, NUM_EPISODES, PRETRAIN_STEPS, BATCH_SIZE, trajectory_len, state_space, action_space, hidden_dim, learning_rate, GAMMA, epsilon, memory, render = False):
    dqfd_agent = agent(state_space, action_space, trajectory_len, hidden_dim, learning_rate, GAMMA, memory)
    
    checkpoint_path = "./training/my_checkpoint"
    
    try:
      dqfd_agent.load_weights(checkpoint_path)
      print("Weights loaded!")
      
    except:
      print('Could not load weights..')
    
    # test(dqfd_agent,env,state_space,action_space,NUM_EPISODES)
    dqfd_agent.pretrain(PRETRAIN_STEPS,BATCH_SIZE)
    print('Pretraining Done!')
    scores = []
    steps = 0
    for ep in range(NUM_EPISODES):
      
      if render:
        env.render()
      state = env.reset()
      score = 0
      n_step_reward = None
      avg_loss = 0
      ep_steps = 0
      while(True):
        
        if np.random.rand() > epsilon:
          action = dqfd_agent.choose_action(tf.reshape(state,[1,state_space]))
          action = np.argmax(action.numpy())
        else:
          action = np.random.randint(0,action_space)
        next_state,reward,done,_ = env.step(action)
        
        score += reward
 			
 			# reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][2]
        dqfd_agent.add_transition(state,action,reward,done,0)
        loss = dqfd_agent.train(steps,BATCH_SIZE)
        avg_loss += loss
        steps+=1
        ep_steps += 1
        state = next_state
 			
        if done: 
          scores.append(score)
          avg_loss = avg_loss/ep_steps
          dqfd_agent.update_target() 
          print('Episode {}, reward : {}'.format(ep,score))
          
          with train_writer.as_default():
            tf.summary.scalar('reward', score, step=ep)
            tf.summary.scalar('avg loss', avg_loss, step=ep)
          
          dqfd_agent.save_weights(checkpoint_path)
          break




if __name__ == "__main__":
  env = gym.make('LunarLander-v2')
  
  STORE_PATH = './DQFD/TensorBoard'
  
  
  NUM_EPISODES = 500
  PRETRAIN_STEPS = 10
  BATCH_SIZE = 32
  trajectory_len = 10
  hidden_dim = 40
  learning_rate = 0.00025
  GAMMA = 0.99
  epsilon = 0.01
  
  
  state_space = len(env.observation_space.high)
  action_space = env.action_space.n
  
  
  expert_demo_data = np.load('demos.npy', allow_pickle=True)
  expert_demo_data = reward_threshold_subset(expert_demo_data,0)
  # print(expert_demo_data.shape)
  expert_demo_data = processor.process_demo_data(expert_demo_data)
  WINDOW_LENGTH = 1
  memory = PartitionedMemory(1000000, pre_load_data=expert_demo_data, alpha=.6, start_beta=.4, end_beta=.4, window_length=WINDOW_LENGTH)



  train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DQFD_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
  
  run_dqfd(env, NUM_EPISODES, PRETRAIN_STEPS, BATCH_SIZE, trajectory_len, state_space, action_space, hidden_dim, learning_rate, GAMMA, epsilon, memory)




