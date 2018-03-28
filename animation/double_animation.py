import sys

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animation.OpenFaceScripts.pipeline import HappyVidMarker


class ColorAnimator:
    def __init__(self, bottom_coords: np.ndarray,
                 bottom_points_colors,
                 top_x: np.ndarray = None, top_y: np.ndarray = None, im=None):
        """

        :param bottom_coords: n x 2 matrix describing  x,y positions of each point on the bottom graph
        :param bottom_points_colors: List of colors for the coordinates, each list entry specifies the colors for a time point
        :param top_x: X coordinates for top plot, optional
        :param top_y: Y coordinates for top plot, optional
        :param im: Background image for bottom plot, optional
        """
        self.xy_pts = bottom_coords
        self.fig, self.ax = plt.subplots()
        self.top_x = top_x
        self.top_y = top_y
        self.points_colors = bottom_points_colors
        if im is not None:
            self.ax.imshow(im)
        self.activity = self.get_first_activity()
        self.patches = self.ax.scatter(*bottom_coords.T, 
                                       c=self.activity, s=100)

    def get_first_activity(self) -> np.ndarray:
        return np.zeros((self.xy_pts.shape[0],))

    def __call__(self, i) -> tuple:
        self.patches.set_color(self.points_colors[i])
        return self.patches,

    def create_animation(self, out_file_name: str):
        print('Bottom part...')
        anim = FuncAnimation(self.fig, self,
                             frames=np.arange(len(self.points_colors)),
                             interval=100, blit=True)
        anim.save(out_file_name)
        print('Top part...')
        if self.top_x is not None and self.top_y is not None:
            HappyVidMarker.bar_movie(out_file_name,
                                     os.path.dirname(out_file_name), self.top_x, self.top_y,
                                     True)
