"""Track pedestrian and group states within a simulation scene."""
from typing import List

import numpy as np
from pysocialforce.utils import stateutils


class PedState:
    """Track the states of pedestrians and social groups."""
    def __init__(self, state, groups, stopId, pretime, coorpath, initialguidepoint, guidelineIndex, roadPolygon, roadPos, roadcenterPos, minXroad, maxYroad, pixelWidthroad, pixelHeightroad, config):
        self.default_tau = config("tau", 0.5)
        self.step_width = config("step_width", 1)
        self.agent_radius = config("agent_radius", 0.35)
        self.max_speed_multiplier = config("max_speed_multiplier", 1.3)
        
        self.pretime = pretime
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
        # self.stepn = stepn
        
        self.max_speeds = None
        self.initial_speeds = None

        self.ped_states = []
        self.group_states = []
        self.stopId_states = []

        self.update(state, groups, stopId)

    def update(self, state, groups, stopId):
        self.state = state
        self.groups = groups
        self.stopId = stopId

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        tau = self.default_tau * np.ones(state.shape[0])
        if state.shape[1] < 8:
            self._state = np.concatenate((state, np.expand_dims(tau, -1), self.initialguidepoint), axis=-1)
        else:
            self._state = state
        if self.initial_speeds is None:
            self.initial_speeds = self.speeds()
        self.max_speeds = self.max_speed_multiplier * self.initial_speeds
        self.ped_states.append(self._state.copy())

        
    def get_states(self):
        return np.stack(self.ped_states), self.group_states

    def size(self) -> int:
        return self.state.shape[0]

    def pos(self) -> np.ndarray:
        return self.state[:, 0:2]

    def vel(self) -> np.ndarray:
        return self.state[:, 2:4]

    def goal(self) -> np.ndarray:
        return self.state[:, 4:6]
    
    def maskPed(self) -> np.ndarray:
        return self.state[:, 4:6]

    def tau(self):
        return self.state[:, 6:7]

    def target(self):
        return self.state[:, 7:8]
    
    def speeds(self):
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def step(self, force, stepn, groups=None):
        """Move peds according to forces"""
        # identify considering pedestrians
        elapsed_time = (stepn+1)*self.step_width
        time_mask = self.pretime <= elapsed_time
        indices = np.where(time_mask == True)
        # desired velocity
        desired_velocity = self.vel() + self.step_width * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)
        # stop when arrived
        desired_velocity[stateutils.desired_directions(self.state)[1] < 0.5] = [0, 0]

        posevacuee = self.state[:,0:2]
        posshelter = self.state[:,4:6]
        expanded_posevacuee = posevacuee[:, np.newaxis, :]
        expanded_posshelter = posshelter[np.newaxis, :, :]
        distances = np.linalg.norm(expanded_posevacuee - expanded_posshelter, axis=2)
        indicesStop = np.min(distances, axis=1) < 3
        # indicesStop = stateutils.desired_directions(self.state)[1] < 0.5
        next_stopId = self.stopId
        next_stopId[indicesStop] += 1
        
        # update target
        targetLast = self.state[:,7:8]
        # update state
        next_state = self.state
        
        if indices[0].size > 0:
            newpos = next_state[indices[0], 0:2] + desired_velocity[indices[0]] * self.step_width
            
            # Return pedestrians who leave the walkable road area.
            # if newpos[:,0]-self.minXroad >= 0:
            colid = np.minimum(((newpos[:,0]-self.minXroad) // self.pixelWidthroad).astype(int), self.roadPolygon.shape[1]-1)
            # else:
            #     colid = 0
            
            # if self.maxYroad - newpos[:, 1] >= 0:
            rowid = np.minimum(((self.maxYroad - newpos[:, 1]) // -self.pixelHeightroad).astype(int), self.roadPolygon.shape[0]-1)
            # else:
            #     rowid = 0

            roadPolygon = self.roadPolygon
            roadid = roadPolygon[rowid,colid]
            indicesoutroad = np.where(roadid == 0)
            
            # Move off-road pedestrians to the nearest road-centerline point.
            # if stepn < 5 :
            newpos3 = np.expand_dims(newpos, axis=0)
            roadcenterPos = self.roadcenterPos
            roadcenterPos3 = roadcenterPos[:, np.newaxis]
            distance = np.linalg.norm(newpos3-roadcenterPos3, axis=2)
            min_indices = np.argmin(distance, axis=0)
            
            newpos[indicesoutroad[0]] = roadcenterPos[min_indices[indicesoutroad[0]]]
        
            next_state[indices[0], 0:2] = newpos
            next_state[indices[0], 2:4] = desired_velocity[indices[0]]
            next_target = stateutils.target(next_state[:,0:2], targetLast, self.coorpath, self.guidelineIndex)[0]
            next_state[indices[0], 7:8] = next_target[indices[0]].reshape(-1, 1)
        
        # next_state[indices, 4:6] = self.shelterAssemble[next_stopId,np.arange(200), :]
        next_groups = self.groups
        
        if groups is not None:
            next_groups = groups
        self.update(next_state, next_groups, next_stopId)

    # def initial_speeds(self):
    #     return stateutils.speeds(self.ped_states[0])

    def desired_directions(self):
        return stateutils.desired_directions(self.state)[0]

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    @property
    def groups(self) -> List[List]:
        return self._groups

    @groups.setter
    def groups(self, groups: List[List]):
        if groups is None:
            self._groups = []
        else:
            self._groups = groups
        self.group_states.append(self._groups.copy())

    def has_group(self):
        return self.groups is not None

    # def get_group_by_idx(self, index: int) -> np.ndarray:
    #     return self.state[self.groups[index], :]

    def which_group(self, index: int) -> int:
        """find group index from ped index"""
        for i, group in enumerate(self.groups):
            if index in group:
                return i
        return -1

    @property
    def stopId(self):
        return self._stopId

    @stopId.setter
    def stopId(self, stopId):
        self._stopId = stopId
        self.stopId_states.append(self._stopId.copy())

class EnvState:
    """State of the environment obstacles"""

    def __init__(self, obstacles, resolution=10):
        self.resolution = resolution
        self.obstacles = obstacles

    @property
    def obstacles(self) -> List[np.ndarray]:
        """obstacles is a list of np.ndarray"""
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obstacles):
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        if obstacles is None:
            self._obstacles = []
        else:
            self._obstacles = [obstacles]
            # for startx, endx, starty, endy in obstacles:
            #     samples = int(np.linalg.norm((startx - endx, starty - endy)) * self.resolution)
            #     line = np.array(
            #         list(
            #             zip(np.linspace(startx, endx, samples), np.linspace(starty, endy, samples))
            #         )
            #     )
            #     self._obstacles.append(line)
