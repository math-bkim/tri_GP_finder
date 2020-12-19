#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 21:44:24 2020

@author: bkim
"""


import math, random, time
import numpy as np
from fractions import Fraction
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(torch.nn.Module):
    
    def __init__(self, hps):
        super(Policy, self).__init__()
        
        self.hps = hps
        self.hidden = self.hps.hidden
        
        self.fc1 = nn.Linear(2, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 1)
        
    
    def step(self, state, a):
        
        done = False
        n_state = state[:]
        n_state[0] -= Fraction(a, 100)
        n_state[1] += Fraction(1,int(math.sqrt(a*8+1) -1)//2) 
            
        if n_state[0]  < 0.03  or n_state[1] > 1:
          done = True
            
        return n_state, None, done, None
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
    
    def decorder(self, n, ptn, expl_noise, render_op=False):
        s = [ Fraction(n-sum(ptn),100),  sum(Fraction(1,int(math.sqrt(pt*8+1) -1)//2) for pt in ptn)]
        pi =ptn[:]
        if s[0] < 0.03 or s[1] > 1:
            done = True
        else:
            done = False
        
        while not done:
            dist_goal = int(s[0]*100) # distance toward 0
            s = list(map(float, s))
            a = (self.forward(torch.from_numpy( np.array(s)).float()).detach().item()
                 + np.random.normal(0, expl_noise, size=1)
                ).clip(-1, 0.9999)
            a_list = [i*(i+1)//2 for i in range(2, int(math.sqrt(dist_goal))*2+5) if i*(i+1)//2 <= dist_goal]
            a  = int(len(a_list)*(a + 1.)/2)  # change tanh input to index
            pi.append(a_list[a])
            s, _, done, _ = self.step(s, a_list[a])
        
        if render_op:
            print("""
                  targeted number : %s
                  resulting partition : %s
                  its weight  : %s
                  its reciprocal sum : %s
                  """ %(n, pi, sum(pi), sum(Fraction(1,int(math.sqrt(pt*8+1) -1)//2) for pt in pi)))
        else:
            return pi
            
    def load_models(self, name=None):
        if name ==None:
            name = self.hps.load_file_name
        self.load_state_dict(torch.load(name + '_policy.pt', map_location=torch.device('cpu') )
)
        print ('Models loaded succesfully')



class GA:
    def __init__(self, hps, target_n, init_ptn = []):
        
        self.hps = hps
        self.actor = Policy(hps)
        self.actor.load_models()
        self.target_n = target_n
        self.init_ptn = init_ptn
        self.pop_size = hps.pop_size
        self.mu_rate  =  hps.mu_rate
        self.pop = []
        self.init()
        self.best_partition, self.best_fitness = self.best_gene()
        self.output_filename = 'output_for_n=' + str(self.target_n) +'.txt'
        self.f = open(self.output_filename, "a")
        print("""
              Target number  : %d
              desired parts  : %s
              Agent's name : %s
              Initial population : %.1f k
              Initial exploration_rate : %.2f
              Initial best candidate :
              %s 
              Initial fitness :  %.3f
              """ 
              %(self.target_n, self.init_ptn, self.hps.load_file_name, self.pop_size/1000, self.mu_rate, self.best_partition, self.best_fitness),
              file = self.f)
            
        
    def init(self):
        
        for i in range(5):
            self.pop.append(self.actor.decorder(self.target_n, self.init_ptn,.0))
        
        while len(self.pop) < self.pop_size:
            self.pop.append(self.actor.decorder(self.target_n, self.init_ptn, .05))
        pass
    
    def recisum(self, pi):
        return sum(Fraction(1,int(math.sqrt(pt*8+1) -1)//2) for pt in pi)
    
    def fitness(self, pi):
        fit = 200*self.target_n / (abs(self.target_n - sum(pi))+self.target_n)
        fit += 800 /(abs(1-self.recisum(pi))+1)
        return fit
    
    def best_gene(self):
        gene, fit = self.pop[0], self.fitness(self.pop[0])
        for ptn in self.pop[1:]:
          if self.fitness(ptn) > fit:
            gene, fit = ptn[:], self.fitness(ptn)
        return gene, fit
    
    
    
    def mutation(self, ptn):
        
        gene = ptn[len(self.init_ptn):]
        if len(gene)>1:
          num_cut = np.random.randint(1, len(gene))
          gene = random.sample(gene, num_cut)   
        else:
          gene = gene[:1]
        
        gene = self.init_ptn + gene
        gene = self.actor.decorder(self.target_n, gene, self.mu_rate*random.randint(0,10)/10)
        
        return gene

    
    def cross_over(self):
        
        ind = np.random.randint(0, len(self.pop), size = 2)
        parent =[[],[]]
        parent[0], parent[1] = self.pop[ind[0]][:] , self.pop[ind[1]][:]
        ll = min(len(parent[0]), len(parent[1]))
        gene = []
        for i in range(ll):
            gene.append(parent[np.random.randint(0,2)][i])
        if sum(gene)>=self.target_n:
            return self.actor.decorder(self.target_n, gene[:len(self.init_ptn)+1], .0)
        else:
            gene = self.actor.decorder(self.target_n, gene, .0)
            return gene
    
    
    def evolve(self):
      
      new_pop = []
      new_pop.append(self.best_gene()[0])
      for i in range(100):
        new_pop.append(self.mutation(new_pop[0]))
        new_pop.append(self.cross_over())
      for ptn in self.pop[1:]:
        new_pop.append(self.mutation(ptn))
      
            
      return new_pop

    def evolve2(self):
      
      new_pop = []
      new_pop.append(self.best_gene()[0])
      
      for i in range(len(self.pop)-1):
        coin = random.random()
        temp = self.cross_over()
        if coin > 0.95:
          temp = self.mutation(temp)
        new_pop.append(temp)
            
      return new_pop
    
    
    def run(self, upper_bound=200):
        
        u_ctr = 0
        for n_gen in range(upper_bound):
            
            if u_ctr >= 30:
              u_ctr = 0
              if self.mu_rate<.41:
                self.mu_rate +=0.05
                print("exploration_rate is now %.2f" %self.mu_rate)
              else:
                break
              
            if n_gen %2 ==0:
              self.pop = self.evolve()
            else:
              self.pop =self.evolve2()

            u_ctr +=1
            best_partition, best_fitness = self.best_gene()
            if best_fitness > self.best_fitness:
                u_ctr = 0
                self.mu_rate = self.hps.mu_rate
                self.best_partition = best_partition[:]
                self.best_fitness = best_fitness
                print("improved.")
                print("%d-th generation" %(n_gen+1), file=self.f )
                print(""" best candidate : %s,  
                          its fitness : %.3f
                          its weight : %d
                          its reciprocal sum : %s
                          """ 
                         %(self.best_partition, self.best_fitness , sum(self.best_partition), self.recisum(self.best_partition)) 
                         , file=self.f)
                print("exploration rate is now %.2f" %self.mu_rate)
              
                    
            if self.best_fitness ==1000.0:
                print("success!")
                print("success!", file=self.f)
                
                found = self.best_partition
                found.sort(reverse=True)
                print("""
                      A desired partition is found at %d-th generation
                      found partition is
                      %s
                      weight : %d
                      reciprocal sum of parts : %s
                          
                      """
                      %(n_gen+1, found, sum(found), sum(Fraction(1,int(math.sqrt(pt*8+1) -1)//2) for pt in found))
                      , file = self.f)
                break
            
            if (n_gen+1)%10 ==0:
                print("%d-th generation is over" %(n_gen+1))
                print("Current population is %.1f k"  %(len(self.pop)/1000) )
                self.f.close()
                self.f = open(self.output_filename, "a")
                time.sleep(self.hps.rest_time)
                
        print("done.", file=self.f)
        print("done.")
        self.f.close() 
        pass
        


class hp:   #collection of hyper parameters#
    def __init__(self):
        
        self.load_file_name = 'agent1'
        self.hidden = 300
        self.pop_size = int(5e2)
        self.mu_rate = 0.1
        self.rest_time = 15




def main(target_n, init_ptn, run_limit):
    
    hps = hp()
    ga = GA(hps, target_n, init_ptn)
    ga.run(run_limit)

if __name__=='__main__':
    
    main(1728, [], 250)

    

