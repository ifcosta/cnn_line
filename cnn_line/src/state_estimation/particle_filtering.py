#!/usr/bin/env python

import numpy as np
from filterpy.monte_carlo import residual_resample


class ParticleFilter():
    def __init__(self, fitness, transition, nfeatures=5,nparticles=100, distribution=None, mu = None, sigma = None, lb = None, ub = None):
        self.fitness = fitness
        self.transition = transition
        self.nfeatures = nfeatures
        self.nparticles = nparticles
        self.distribution = distribution
        self.mu = mu
        self.sigma = sigma
        self.lb = lb
        self.ub = ub
        self.initialize_particles()
    

    def initialize_particles(self):
        self.particles = []
        if(self.distribution):
            for i in range(self.nfeatures):
                if(self.distribution[i] == 'normal'):
                    self.particles.append(np.random.normal(size=(self.nparticles, 1), loc = self.mu[i], scale=self.sigma[i]))
                else:
                    self.particles.append(np.random.uniform(low = self.lb[i], high = self.ub[i], size=(self.nparticles, 1)))
        else:
            self.particles = np.random.normal(size=(self.nparticles, self.nfeatures))
            '''if(self.sigma): 
                self.particles *= self.sigma
            if(self.mu):
                self.particles += self.mu'''
            
        self.particles = np.array(self.particles).reshape(self.nfeatures,self.nparticles)
        self.particles = np.swapaxes(self.particles, 0, 1)
        if(self.lb):
            self.particles = np.clip(self.particles, self.lb, np.inf)
        if(self.ub):
            self.particles = np.clip(self.particles, -np.inf, self.ub)
        self.weights = np.zeros(shape=(self.nparticles))

    
    def return_particles(self):
        return self.particles
    def return_20_mean(self):
        idx = np.argpartition(self.weights, int(self.nparticles * 0.8))

        return np.average(self.particles[idx[int(self.nparticles * 0.8):]], weights = self.weights[idx[int(self.nparticles * 0.8):]], axis = 0)
    def return_best_mean(self, value = 0.95):
        idx = np.argpartition(self.weights, int(self.nparticles * value))
        #plt.plot(self.weights[idx[int(self.nparticles * 0.95):]])
        
        return np.average(self.particles[idx[int(self.nparticles * value):]], weights = self.weights[idx[int(self.nparticles * value):]], axis = 0)

    def return_mean(self):
        return np.mean(self.particles, axis= 0)

    def predict(self, u):
        self.particles = self.transition(self.particles, u)
        self.clip_particles()



    def update(self, observation):
        self.weights = self.fitness(self.particles, observation)
        self.weights += 1.e-20
        self.weights /= np.sum(self.weights)
    
    def resample(self):
        indexes = residual_resample(self.weights)
        self.particles = self.particles[indexes,:]
        self.weights = self.weights[indexes]

    
    def clip_particles(self):
        for i in range(self.nfeatures):
            self.particles[:,i] = np.clip(self.particles[:,i], self.lb[i], self.ub[i])

    


    def next_step(self, observation, u):
        self.predict(u)
        self.update(observation)
        self.resample()
        




