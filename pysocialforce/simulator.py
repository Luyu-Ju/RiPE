# coding=utf-8

"""Synthetic pedestrian behavior with social groups simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""
from pysocialforce.utils import DefaultConfig
from pysocialforce.scene import PedState, EnvState
from pysocialforce import forces
import numpy as np

class Simulator:
    """Simulate social force model.

    ...

    Attributes
    ----------
    state : np.ndarray [n, 6] or [n, 7]
       Each entry represents a pedestrian state, (x, y, v_x, v_y, d_x, d_y, [tau])
    obstacles : np.ndarray
        Environmental obstacles
    groups : List of Lists
        Group members are denoted by their indices in the state
    config : Dict
        Loaded from a toml config file
    max_speeds : np.ndarray
        Maximum speed of pedestrians
    forces : List
        Forces to factor in during navigation

    Methods
    ---------
    capped_velocity(desired_velcity)
        Scale down a desired velocity to its capped speed
    step()
        Make one step
    """
    
    def __init__(self, state, stopId, groups=None, obstacles=None, pretime=None, coorpath=None, initialguidepoint=None, guidelineIndex=None, roadPolygon=None, roadPos=None, roadcenterPos=None, minXroad=None, maxYroad=None, pixelWidthroad=None, pixelHeightroad=None, config_file=None):
        self.config = DefaultConfig()
        if config_file:
            self.config.load_config(config_file)
        # TODO: load obstacles from config
        self.scene_config = self.config.sub_config("scene")
        
        self.pretime = pretime    #  self.pretime = pretime - np.min(pretime)
        
        self.initialtime = np.min(pretime)
        
        self.coorpath = coorpath
        
        self.initialguidepoint = initialguidepoint
        self.guidelineIndex = guidelineIndex
        self.roadPolygon = roadPolygon
        self.roadPos = roadPos
        self.roadcenterPos = roadcenterPos
        self.minXroad = minXroad
        self.maxYroad = maxYroad
        self.pixelWidthroad = pixelWidthroad
        self.pixelHeightroad = pixelHeightroad
        # self.stepn = []
        # self.boundary1 = boundary1
        # self.boundary2 = boundary2
        
        # initiate obstacles
        self.env = EnvState(obstacles, self.config("resolution", 10.0))

        # initiate agents
        self.peds = PedState(state, groups, stopId, self.pretime, self.coorpath, self.initialguidepoint, self.guidelineIndex, self.roadPolygon, self.roadPos, self.roadcenterPos, self.minXroad, self.maxYroad, self.pixelWidthroad, self.pixelHeightroad, self.config)               

        # construct forces
        self.forces = self.make_forces(self.config)
        

    def make_forces(self, force_configs):
        """Construct forces"""
        force_list = [
            forces.DesiredForce(),
            forces.SocialForce(),
            forces.ObstacleForce(),
            # forces.PedRepulsiveForce(),
            # forces.SpaceRepulsiveForce(),
        ]
        group_forces = [
            forces.GroupCoherenceForceAlt(),
            forces.GroupRepulsiveForce(),
            forces.GroupGazeForceAlt(),
        ]
        if self.scene_config("enable_group"):
            force_list += group_forces

        # initiate forces
        for force in force_list:
            force.init(self, force_configs)

        return force_list

    def compute_forces(self, stepn):
        """compute forces"""
        return sum(map(lambda x: x.get_force(stepn), self.forces))

    def get_states(self):
        """Expose whole state"""
        return self.peds.get_states()

    def get_length(self):
        """Get simulation length"""
        return len(self.get_states()[0])

    def get_obstacles(self):
        return self.env.obstacles

    def step_once(self, stepn):
        """step once"""
        self.peds.step(self.compute_forces(stepn), stepn)

    def step(self, n=1):
        """Step n time"""
        for stepn in range(n):
            # self.stepn = stepn
            # self.peds = PedState(self.peds.state, self.peds.groups, self.pretime, self.stepn, self.config)
            # stepn = 100000
            self.step_once(stepn)
        return self
    
