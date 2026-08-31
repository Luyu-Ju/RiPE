"""Utility functions for plots and animations."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import geopandas as gpd

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpl_animation
    from matplotlib.patches import Circle, Polygon
    from matplotlib.collections import PatchCollection
except ImportError:
    plt = None
    mpl_animation = None

from .logging import logger


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"


@contextmanager
def canvas(image_file=None, **kwargs):
    """Generic matplotlib context."""
    fig, ax = plt.subplots(**kwargs)
    ax.grid(linestyle="dotted")
    ax.set_aspect(1.0, "datalim")
    ax.set_axisbelow(True)

    yield ax

    fig.set_tight_layout(True)
    if image_file:
        fig.savefig(image_file, dpi=300)
    # fig.show()
    plt.close(fig)


@contextmanager
def animation(length: int, movie_file=None, writer=None, **kwargs):
    """Context for animations."""
    fig, ax = plt.subplots(**kwargs)
    fig.set_tight_layout(True)
    ax.grid(linestyle="dotted")
    ax.set_aspect(1.0, "datalim")
    ax.set_axisbelow(True)

    context = {"ax": ax, "update_function": None, "init_function": None}
    yield context

    ani = mpl_animation.FuncAnimation(
        fig,
        init_func=context["init_function"],
        func=context["update_function"],
        frames=length,
        blit=True,
    )
    if movie_file:
        ani.save(movie_file, writer=writer)
    # fig.show()
    plt.close(fig)


class SceneVisualizer:
    """Visualize pedestrian states and animate a simulation scene."""

    def __init__(
        self, scene, output=None, writer="imagemagick", cmap="viridis", agent_colors=None, **kwargs
    ):
        self.scene = scene
        self.states, self.group_states = self.scene.get_states()
        
        #
        sort_initime = np.array(sorted(set(self.scene.pretime)))
        initIndex = (sort_initime//self.scene.peds.step_width).astype(int)
        A = self.states
        group_A = self.group_states
        # for i in range(len(initIndex)):
        #     B = self.states[initIndex[i]:initIndex[i]+self.states.shape[0],:,:]   # 25
        #     A = np.concatenate((A, B), axis=0)
        #     group_B = self.group_states[initIndex[i]:initIndex[i]+self.states.shape[0]]   # 25
        #     group_A += group_B
        self.states_plot = A
        self.group_states_plot = group_A
        # sort_initime = np.array(sorted(set(self.scene.pretime)))
        # initIndex = (sort_initime//1).astype(int)  # change
        # A = np.zeros((1, self.states.shape[1], self.states.shape[2]))
        # group_A = [np.random.randint(low=0, high=10)]
        # for i in range(len(initIndex)):
        #     B = self.states[initIndex[i]:initIndex[i]+25,:,:]   # 25
        #     A = np.concatenate((A, B), axis=0)
        #     group_B = self.group_states[initIndex[i]:initIndex[i]+25]   # 25
        #     group_A += group_B
        # self.states_plot = A[1:,:,:]
        # self.group_states_plot = group_A[1:]
        #
        
        self.cmap = cmap
        self.agent_colors = agent_colors
        self.frames = self.states_plot.shape[0]  # self.frames = self.scene.get_length()       # change
        self.output = output
        self.writer = writer

        self.fig, self.ax = plt.subplots(**kwargs)
        
        x_ticks = [841440, 841480, 841520, 841560, 841600]  # Map-coordinate tick positions.
        x_labels = ['841440', '841480', '841520', '841560', '841600']  # Displayed tick labels.
        plt.xticks(x_ticks, x_labels)

        boundpath = DATA_DIR / "Road" / "Boundary_Graph.shp"
        gdfpath = gpd.read_file(boundpath)
        gdfpath.plot(ax=self.ax, color='black', linewidth=1)
        
        buildingpath = DATA_DIR / "Building" / "Buildings.shp"
        gdfbuilding = gpd.read_file(buildingpath)
        gdfbuilding.plot(ax=self.ax, facecolor='0.8', edgecolor='none')
        # gdfbuilding.plot(ax=self.ax, facecolor='none', edgecolor='red', linewidth=1, linestyle='dashed')
        
        landslidepath = DATA_DIR / "Landslide" / "Landslide.shp"
        gdflandslide = gpd.read_file(landslidepath)
        gdflandslide.plot(ax=self.ax, facecolor='none', edgecolor='gray',linestyle='--')

        flowdirectionpath = DATA_DIR / "Landslide" / "Flow direction.shp"
        gdfflowdirection = gpd.read_file(flowdirectionpath)
        gdfflowdirection.plot(ax=self.ax, color='gray',linewidth=1)
        
        flowarrowpath = DATA_DIR / "Landslide" / "Arrow.shp"
        gdfflowarrow = gpd.read_file(flowarrowpath)
        gdfflowarrow.plot(ax=self.ax, facecolor='gray', edgecolor='none')
        
        sheltersignpath = DATA_DIR / "Shelter" / "Sign.shp"
        gdfsheltersign = gpd.read_file(sheltersignpath)
        gdfsheltersign.plot(ax=self.ax, color='gray',linewidth=1)
        
        sheltersignInpath = DATA_DIR / "Shelter" / "SignIn.shp"
        gdfsheltersignIn = gpd.read_file(sheltersignInpath)
        gdfsheltersignIn.plot(ax=self.ax, facecolor='0.8', edgecolor='none')
        
        self.time_text = self.ax.text(0.85, 0.95, '', transform=self.ax.transAxes, ha='right', fontsize=10)

        self.ani = None

        self.group_actors = None
        self.group_collection = PatchCollection([])
        self.group_collection.set(
            animated=True,
            alpha=0.2,
            cmap=self.cmap,
            facecolors="none",
            edgecolors="purple",
            linewidth=2,
            clip_on=True,
        )

        self.human_actors = None
        self.human_collection = PatchCollection([])
        self.human_collection.set(animated=True, alpha=0.6, cmap=self.cmap, clip_on=True)

    def plot(self):
        """Main method for create plot"""
        # self.plot_obstacles()
        # self.plot_boundary()

        groups = self.group_states[0]  # static group for now
        if not groups:
            for ped in range(self.scene.peds.size()):
                x = self.states[:, ped, 0]
                y = self.states[:, ped, 1]
                self.ax.plot(x, y, "-o", label=f"ped {ped}", markersize=4)
        else:

            colors = plt.cm.rainbow(np.linspace(0, 1, len(groups)))

            for i, group in enumerate(groups):
                for ped in group:
                    x = self.states[:, ped, 0]
                    y = self.states[:, ped, 1]
                    self.ax.plot(x, y, "-o", label=f"ped {ped}", markersize=4, color=colors[i])
        self.ax.legend()
        return self.fig

    def animate(self):
        """Main method to create animation"""

        self.ani = mpl_animation.FuncAnimation(
            self.fig,
            init_func=self.animation_init,
            func=self.animation_update,
            frames=self.frames,
            blit=True,
        )

        return self.ani

    def __enter__(self):
        logger.info("Start plotting.")
        self.fig.set_tight_layout(True)
        self.ax.grid(linestyle="dotted")
        self.ax.set_aspect("equal")
        self.ax.margins(2.0)
        self.ax.set_axisbelow(True)
        self.ax.set_xlabel("Latitude (m)", fontsize=10)
        self.ax.set_ylabel("Longitude (m)", fontsize=10)
        self.ax.tick_params(axis='x', labelsize=10)
        self.ax.tick_params(axis='y', labelsize=10)
        self.ax.text(0.75, 0.85, 'Landslide', transform=self.ax.transAxes, ha='right', fontsize=10)
        self.ax.text(0.18, 0.4, 'Shelter', transform=self.ax.transAxes, ha='right', fontsize=10)

        plt.rcParams["animation.html"] = "jshtml"

        # x, y limit from states, only for animation
        # margin = 20
        # xy_limits = np.array(
        #     [minmax(state) for state in self.states]
        # )  # (x_min, y_min, x_max, y_max)
        # xy_min = np.min(xy_limits[:, :2], axis=0) - margin
        # xy_max = np.max(xy_limits[:, 2:4], axis=0) + margin
        self.ax.set(xlim=(841440, 841600), ylim=(818050, 818230))

        # # recompute the ax.dataLim
        # self.ax.relim()
        # # update ax.viewLim using the new dataLim
        # self.ax.autoscale_view()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type:
            logger.error(
                f"Exception type: {exception_type}; Exception value: {exception_value}; Traceback: {traceback}"
            )
        logger.info("Plotting ends.")
        if self.output:
            if self.ani:
                output = self.output + ".gif"
                logger.info(f"Saving animation as {output}")
                self.ani.save(output, writer=self.writer)
            else:
                output = self.output + ".png"
                logger.info(f"Saving plot as {output}")
                self.fig.savefig(output, dpi=300)
        plt.close(self.fig)

    def plot_human(self, step=-1):
        """Generate patches for human
        :param step: index of state, default is the latest
        :return: list of patches
        """
        # states, _ = self.scene.get_states()       # change
        current_state = self.states_plot[step]
        # radius = 0.2 + np.linalg.norm(current_state[:, 2:4], axis=-1) / 2.0 * 0.3
        radius = [0.8] * current_state.shape[0]
        if self.human_actors:
            for i, human in enumerate(self.human_actors):
                human.center = current_state[i, :2]
                human.set_radius(0.8)
                # human.set_radius(radius[i])
        else:
            self.human_actors = [
                Circle(pos, radius=r) for pos, r in zip(current_state[:, :2], radius)
            ]
        self.human_collection.set_paths(self.human_actors)
        if not self.agent_colors:
            self.human_collection.set_array(np.arange(current_state.shape[0]))
        else:
            # set colors for each agent
            assert len(self.human_actors) == len(
                self.agent_colors
            ), "agent_colors must be the same length as the agents"
            self.human_collection.set_facecolor(self.agent_colors)

    def plot_groups(self, step=-1):
        """Generate patches for groups
        :param step: index of state, default is the latest
        :return: list of patches
        """
        # states, group_states = self.scene.get_states()    # change
        current_state = self.states_plot[step]
        current_groups = self.group_states_plot[step]
        if self.group_actors:  # update patches, else create
            points = [current_state[g, :2] for g in current_groups]
            for i, p in enumerate(points):
                self.group_actors[i].set_xy(p)
        else:
            self.group_actors = [Polygon(current_state[g, :2]) for g in current_groups]

        self.group_collection.set_paths(self.group_actors)

    # def plot_obstacles(self):
    #     obsplot = self.scene.get_obstacles()
    #     obsplot = obsplot[0]
    #     # obsplotx = obsplot[:,0]
    #     # obsploty = obsplot[:,1]
    #     for s in obsplot:
    #         self.ax.plot(s[0], s[1], "-o", color="black", markersize=0.3)

    # def plot_boundary(self):
    #     bound1plot = self.scene.boundary1
    #     bound2plot = self.scene.boundary2
        
    #     x1 = bound1plot[:, 0]
    #     y1 = bound1plot[:, 1]
        
    #     x2 = bound2plot[:, 0]
    #     y2 = bound2plot[:, 1]
        
    #     self.ax.plot(x1, y1, color="black", linestyle='-')
    #     self.ax.plot(x2, y2, color="black", linestyle='-')
        
    def animation_init(self):
        # self.plot_obstacles()
        # self.plot_boundary()
        self.ax.add_collection(self.group_collection)
        self.ax.add_collection(self.human_collection)

        return (self.group_collection, self.human_collection, self.time_text)

    def animation_update(self, i):
        self.plot_groups(i)
        self.plot_human(i)
        sort_initime = np.array(sorted(set(self.scene.pretime)))
        # t = (sort_initime[i // 25] + self.scene.initialtime + self.scene.peds.step_width * (i % 25)) // 60 # 500
        t = i
        self.time_text.set_text('Time: {:.1f} s'.format(t))

            
        # sort_initime = sorted(set(self.scene.pretime))
        # elapsed_time = (i+1)*self.scene.peds.step_width
        # diff_time = elapsed_time - np.array(sort_initime)
        # diff_time[diff_time < 0] = 99999999
        # min_value = np.min(diff_time)
        # # min_index = np.argmin(diff_time)
        
        # if min_value <= 1:
        #     for z in range(100):
        #         zz = z + i
        #         self.plot_groups(zz)
        #         self.plot_human(zz)
        #         t = ((zz)*self.scene.peds.step_width + self.scene.initialtime) // 60
        #         self.time_text.set_text('Time: {:.1f} min'.format(t))
        return (self.group_collection, self.human_collection, self.time_text)
