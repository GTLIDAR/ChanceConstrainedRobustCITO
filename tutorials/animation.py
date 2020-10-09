"""
Name:           Animations.py
Author:         Luke Drnach
Version:        1
Date:           September 16, 2019
Description:    Example script on making animations with matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

# Create a new figure with axis and plot to animate
fig = plt.figure()
ax = plt.axes(xlim=(0,2), ylim=(-2,2))
line, = ax.plot([], [], lw=2)

# Initialization function plots the background of each frame
def init():
    line.set_data([], [])
    return line, 

# Animation function updates the plot
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line, 

# Call the animator. blit=True only re-draws the parts that have changed
anim = animation.FuncAnimation(fig, animate, init_func=init,
 frames=200, interval=20, blit=True)

# Optional save function. Requires ffmpeg or mencoder to be installed
# anim.save('basic_animation.mp4',fps=30,extra_args=['-vcodec','libx264'])

# Show the animation
plt.show()

# Print
print('Finished')