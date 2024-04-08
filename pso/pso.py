import numpy as np
from optim.rastrigin import rastrigin

class Particle:
    def __init__(self, env_dim):
        self.position = np.random.uniform(-5.12, 5.12, env_dim)
        self.velocity = np.zeros(env_dim)
        self.best_fitness = 100000
        self.update_fitness()
        self.best_position = self.position

    def update_fitness(self):
        current_fitness = rastrigin(self.position)
        if self.best_fitness > current_fitness:
            self.best_fitness = current_fitness
            self.best_position = self.position
        
        self.fitness = current_fitness

    def update_velocity(self, cfg, r_p, r_g, swarm):
        self.velocity = cfg.w * self.velocity + cfg.phi_p * r_p * (self.best_position - self.position) + cfg.phi_g * r_g * (swarm.best_swarm_position - self.position)

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, -5.12, 5.12)

class Swarm:
    def __init__(self, env_dim, num_particles):
        self.env_dim = env_dim
        self.num_particles = num_particles
        self.population = []
        self.best_swarm_fitness = 1000000

        self.init_population()

    def init_population(self):
        for i in range(self.num_particles):
            self.population.append(Particle(env_dim=self.env_dim))
            if self.population[i].fitness < self.best_swarm_fitness:
                self.best_swarm_fitness = self.population[i].fitness
                self.best_swarm_position = self.population[i].position

    def update_best_fitness(self, fitness, position):
        self.best_swarm_fitness = fitness
        self.best_swarm_position = position

