# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pose tasks with DClaw robots.

The DClaw is tasked to match a pose defined by the environment.
"""

import abc
import collections
from typing import Any, Dict, Optional, Sequence
import time
import numpy as np

from robel.components.robot.dynamixel_robot import DynamixelRobotState
from robel.components.robot import RobotComponentBuilder, RobotState
from robel.dhand.base_env import BaseDHandObjectEnv
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.resources import get_asset_path
from robel.dclaw import scripted_reset
from robel.components.robot.dynamixel_utils import CalibrationMap


# The observation keys that are concatenated as the environment observation.
DEFAULT_OBSERVATION_KEYS = (
    'qpos',
    #'qpos_error',
)

# The maximum velocity for the motion task.
MOTION_VELOCITY_LIMIT = np.pi / 6  # 30deg/s

# The error margin to the desired positions to consider as successful.
SUCCESS_THRESHOLD = 10 * np.pi / 180

DCLAW3_ASSET_PATH = 'robel-scenes/dhand/dhand.xml'


class BaseDHandReposition(BaseDHandObjectEnv, metaclass=abc.ABCMeta):
    """Shared logic for DClaw pose tasks."""

    def __init__(self,
                 asset_path: str = DCLAW3_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 frame_skip: int = 50, # 10Hz, dt = 0.002
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            frame_skip: The number of simulation steps per environment step.
        """
        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)
        self._initial_pos = np.zeros(23)
        self._desired_pos = np.zeros(23)

    def _configure_robot(self, builder: RobotComponentBuilder):
        super()._configure_robot(builder)
        # Add an overlay group to show desired joint positions.
        builder.add_group(
            'overlay', actuator_indices=[], qpos_indices=range(16))

    def _configure_object(self, builder2):
        builder2.add_group(
            'object',
            qpos_indices=range(2),
            qpos_range=[
                (-1e8, 1e8)]*2,
            qvel_range=None,
            actuator_indices=None
            )
        CALIBRATION_MAP = CalibrationMap({10: (1, -3.1415), 20: (1, -3.1415)})
        builder2.set_dynamixel_device_path('/dev/ttyUSB1')
        builder2.set_hardware_calibration_map(CALIBRATION_MAP)
        builder2.update_group(
                'object', motor_ids=[10, 20])
        #scripted_reset.add_groups_for_reset(builder2)

    def _reset(self):
        """Resets the environment."""
        # Mark the target position in sim.
        self.robot.set_state({
            'dhand': RobotState(qpos=self._initial_pos[7:], qvel=np.zeros(16)),
        })
        init_pos = np.array([3.46,-1.475,-0.54,2.14,-0.15,-0.74,-0.5])
        self.command_sawyer(RobotState(qpos=init_pos, qvel=np.zeros(7)), angles=True)
        
        time.sleep(2)
        
        self.object.set_motors_engaged('object', True)
        CURRENT_LIM = 10
        qp = self.object._get_group_states([self.object.get_config('object')])[0].qpos
        qp0 = qp.copy()

        def get_spool_state():
            return self.object._get_group_states([self.object.get_config('object')])[0]

        last_qpos = None
        done = [False, False]
        while not all(done):
        # while last_qpos is None or not np.allclose(get_spool_state().qpos, last_qpos):
            cur_state = get_spool_state()
            cur = cur_state.current
            if cur[0] < CURRENT_LIM and not done[0]:
                qp[0] += 1
            else:
                done[0] = True
            if cur[1] < CURRENT_LIM and not done[1]:
                qp[1] += 1
            else:
                done[1] = True

            last_qpos = cur_state.qpos
            #import ipdb; ipdb.set_trace()
            self.object.set_state({'object' : RobotState(qpos=qp, qvel=np.zeros(2))})
            #self.object.set_state({'object' : RobotState(qpos=None, qvel=0.3)})
            # qp = qp + 1
        unspool_pos = np.array([41,13])


        step_size = 0.5
        while not np.allclose(get_spool_state().qpos, unspool_pos, atol=2):
            spool_qp = get_spool_state().qpos
            diff = spool_qp - unspool_pos
            if not np.isclose(diff[0], 0, atol=2):
                # Stop changing the 1st motor qpos
                spool_qp[0] -= step_size
            if not np.isclose(diff[1], 0, atol=2):
                # Stop changing the 2nd motor qpos
                spool_qp[1] -= step_size
            self.object.set_state({'object' : RobotState(qpos=spool_qp, qvel=np.zeros(2))})

        self.object.set_motors_engaged('object', False)
        time.sleep(2)

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        #action[:7] are Pose values [px,py,pz,ox,oy,oz,ow]
	    #action[7:] are dhand actions
        self.robot.step({'dhand': action[7:]})
        # Stepping the sawyer
        action_sawyer = np.clip(action[:7], -1, 1)
        #action_sawyer[3:] = np.array([0.586577800386, 0.468474469068, 0.421020306894, 0.50911693854])
        self.command_sawyer(RobotState(qpos=action_sawyer, qvel=np.zeros(7)))

    def get_obs_dict(self) -> Dict[str, Any]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        state = self.robot.get_state('dhand')
        obj_qpos = self.listener.obj_qp
        sqp = self.listener.sawyer_qp
        sqpa = self.listener.sawyer_qpa
        sqp = sqp + sqpa
        # Why are we even doing this??
        self.robot.set_state({'sawyer': RobotState(qpos=sqp, qvel=np.zeros(7))})
        total_state = np.append(sqp, state.qpos, axis=0)
        obj_state = self.object.get_state('object').qpos

        goal_qpos = [0,0,0]

        obs_dict = collections.OrderedDict((
            ('qpos', total_state),
            ('pos_error', np.array(obj_qpos) - np.array(goal_qpos)),
        ))
        # Add hardware-specific state if present.
        if isinstance(state, DynamixelRobotState):
            obs_dict['current'] = state.current

        return obs_dict

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        #qvel = obs_dict['qvel']

        reward_dict = collections.OrderedDict((
            ('pose_error_cost', np.linalg.norm(obs_dict['pos_error'])),
            # Penalty if the velocity exceeds a threshold.
            #('joint_vel_cost',
            # -0.1 * np.linalg.norm(qvel[np.abs(qvel) >= np.pi])),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        mean_pos_error = np.mean(np.abs(obs_dict['pos_error']), axis=1)
        score_dict = collections.OrderedDict((
            # Clip and normalize error to 45 degrees.
            ('points', 1.0 - np.minimum(mean_pos_error / (np.pi / 4), 1)),
            ('success', mean_pos_error < SUCCESS_THRESHOLD),
        ))
        return score_dict

    def _update_overlay(self):
        """Updates the overlay in simulation to show the desired pose."""
        self.robot.set_state({'overlay': RobotState(qpos=self._desired_pos[7:])})

    def _make_random_pose(self) -> np.ndarray:
        """Returns a random pose."""
        pos_range = self.robot.get_config('dhand').qpos_range
        random_range = self.sawyer_bounds
        random_range = np.append(random_range,pos_range.copy(), axis=0)

        # Clamp middle joints to at most 0 (joints always go outwards) to avoid
        # entanglement.
        #random_range[[1, 4, 7], 1] = 0
        pose = self.np_random.uniform(
            low=random_range[:, 0], high=random_range[:, 1])
        #print(pose)
        return pose


@configurable(pickleable=True)
class DHandRepositionFixed(BaseDHandReposition):
    """Track a fixed random initial and final pose."""

    def _reset(self):
        self._initial_pos = np.zeros_like(self._make_random_pose())
        self._update_overlay()
        super()._reset()


if __name__ == "__main__":
    import IPython
    IPython.embed()

