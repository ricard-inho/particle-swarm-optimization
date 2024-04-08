import numpy as np
import matplotlib.pyplot as plt

import imageio
import io

from optim.rastrigin import rastrigin

def visualize(generation_i, swarm, images):
    # Create a meshgrid for visualization
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y])
    plt.imshow(Z, cmap='viridis', extent=[-5.12, 5.12, -5.12, 5.12], vmin=np.min(Z), vmax=np.max(Z))
    
    # Plot each particle position
    for particle in swarm.population:
        plt.scatter(particle.position[0], particle.position[1], color='red')

    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    images.append(imageio.imread(buf))
    plt.close()

def visualize_3D(generation_i, swarm, images):
    # Create a meshgrid for visualization
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Plot each particle position
    for particle in swarm.population:
        ax.scatter(particle.position[0], particle.position[1], rastrigin(particle.position), color='red')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    images.append(imageio.imread(buf))
    plt.close()