import hydra
import imageio
import numpy as np

from utils.visualization import visualize
from pso.pso import Swarm

@hydra.main(version_base=None, config_path="configs", config_name="pso_config")
def main(cfg):
    swarm = Swarm(env_dim=cfg.env_dim, num_particles=cfg.num_particles)

    images = []
    for i in range(cfg.num_iterations):

        visualize(generation_i=i, swarm=swarm, images=images)
        # visualize_3D(generation_i=i, swarm=swarm)

        for particle in swarm.population:
            r_p = np.random.uniform(0,1, cfg.env_dim)
            r_g = np.random.uniform(0,1, cfg.env_dim)

            particle.update_velocity(cfg, r_p, r_g, swarm)
            particle.update_position()
            particle.update_fitness()

            if particle.fitness < swarm.best_swarm_fitness:
                swarm.update_best_fitness(particle.fitness, particle.position)

        print(f"Generation {i} swarm fitness: {swarm.best_swarm_fitness}, swarm position: {swarm.best_swarm_position}")

    imageio.mimsave(cfg.output_folder, images, duration=0.5)


if __name__ == "__main__":
    main()