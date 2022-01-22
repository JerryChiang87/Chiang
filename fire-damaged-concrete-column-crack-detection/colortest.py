from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 50)
h = np.linspace(0, 500, 500)
x = np.outer(512 * np.sin(u), np.ones(len(h)))
y = np.outer(512 * np.cos(u), np.ones(len(h)))
z = np.outer(np.ones(len(u)), h)
# Plot the surface
ax.scatter(269.5, 215.5, 450, c="red")
ax.scatter(269.5, 215.5, 1, c="black", marker="|", linewidths=1.5)
ax.plot_surface(x, y, z, cmap=plt.get_cmap('bone'), alpha=0.3)
plt.show()
