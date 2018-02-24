import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ColorAnimator:
    def __init__(self, bottom_times: np.ndarray, bottom_coords: np.ndarray,
                 points_colors,
                 top_x: np.ndarray=None, top_y: np.ndarray=None, im=None, cmap:
                 str='coolwarm',
                 norm=None):
        xy_pts = bottom_coords
        self.bottom_times = bottom_times
        self.fig, self.ax = plt.subplots()
        self.top_x = top_x
        self.top_y = top_y
        if im:
            self.ax.imshow(im)
        self.activity = self.get_first_activity()
        if not norm:
            self.patches = self.ax.scatter(*xy_pts.T, c=self.activity,
                                           s=100, cmap=cmap)
        else:
            self.patches = self.ax.scatter(*xy_pts.T, c=self.activity, 
                                           s=100, cmap=cmap, norm=norm)

    def init(self):
        self.patches.set_color('w')
        return self.patches,
    
    def __call__(self, i):
       self.patches.set_color(points_colors[i])
       

    def create_animation(self, out_file_name: str):
        anim = FuncAnimation(self.fig, self,
                             frames=np.arange(len(self.bottom_times)),
                             init_func=self.init, interval=100, blit=True)
        anim.save(out_file_name)
