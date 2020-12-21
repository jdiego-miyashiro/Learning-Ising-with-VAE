# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class IsingModel:
    def __init__(self,size,iterations,t):
        self.size=size
        self.iterations=iterations
        self.beta=1/t
        self.configuration = 2*np.random.randint(2, size=(self.size,self.size))-1

    def mcmstep(self):
            #Loop with a size equal to spins in the system
            beta= self.beta
            N = self.size
            for i in range(N):
                for j in range(N):
                        #generate integer random number between 0 and N
                        a = np.random.randint(0, N)
                        b = np.random.randint(0, N)
                        ## Perform a change in the system according to monte carlo move rule
                        s =  self.configuration[a, b]
                        #calculate energy cost of this new configuration (the % is for calculation of periodic boundary condition)
                        nb = self.configuration[(a+1)%N,b] + self.configuration[a,(b+1)%N] + self.configuration[(a-1)%N,b] + self.configuration[a,(b-1)%N]
                        cost = 2*s*nb
                        #flip spin or not depending on the cost and its Boltzmann factor
                        ## (acceptance probability is given by Boltzmann factor with beta = 1/kBT)
                        if cost < 0:
                            s = s*(-1)
                        elif rand() < np.exp(-cost*beta):
                            s = s*(-1)
                        self.configuration[a, b] = s


    def plot_configuration(self):
        X, Y = np.meshgrid(range(self.size),range(self.size))
        plt.pcolormesh(X,Y, self.configuration, vmin=-1.0, cmap='RdBu_r')
        plt.title('Ising Model at ', str(self.beta))
        plt.axis('tight')
        plt.show()

    def display(self):
        print(self.configuration)

    def simulate(self):
        for i in range(self.iterations+1):
            self.mcmstep()
        return self.configuration
    def set_beta(self, t):
        self.beta = 1/t
    def set_iterations(self,iter):
        self.iterations=iter
    def set_configuration(self,new_config):
        self.configuration = new_config
    def reboot(self):
        self.configuration = 2*np.random.randint(2, size=(self.size,self.size))-1
    def calc_magnetization(self):
        mag = np.sum(self.configuration)
        return mag


def generate_beta_data(ising_model,t,number_of_images,reboot=False):
    data=[]

    for ith_image in range(number_of_images):
        ising_model.set_beta(t)
        system_data=ising_model.simulate()
        ith_image_data=(system_data.flatten() + 1)/2
        data.append(np.concatenate((np.array([t]),ith_image_data)))

    if reboot:
        ising_model.reboot()
    return pd.DataFrame.from_records(data)

def generate_data(warm_up_iters,isin_size,t_range,t_step,image_number):


    start=t_range[0]
    stop=t_range[1]
    ising_model = IsingModel(isin_size,50*warm_up_iters,start)
    ising_model.simulate()
    ising_model.set_iterations(warm_up_iters)
    beta_data = generate_beta_data(ising_model,start,image_number)
    for temperature in tqdm(np.arange(start+t_step,stop,t_step)):
        ising_model.set_beta(temperature)
        new_data=generate_beta_data(ising_model,temperature,image_number)
        beta_data=pd.concat([beta_data,new_data])
    print(len(beta_data))
    print(beta_data.loc[0])
    beta_data.to_csv("ising_magnetization_data_wm200_img100_new")

def main():
    #ising_model=IsingModel(28,50,0.1)
    #beta_data = generate_beta_data(ising_model,0.1,1000)
    #beta_data.to_csv("ising_new_complete")

    generate_data(200,28,[0.05,5],0.1,100)

    #beta_data.to_csv("ising_beta")


    pass

main()
