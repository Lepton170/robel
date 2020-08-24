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

"""Shared logic for all DClaw environments."""

import abc
import collections
from typing import Dict, Optional, Sequence, Union

import gym
import numpy as np

from robel.components.robot import RobotComponentBuilder, RobotState
from robel.components.robot.dynamixel_utils import CalibrationMap
from robel.dclaw import scripted_reset
from robel.robot_env import make_box_space, RobotEnv

from rospy_tutorials.msg import Floats
from geometry_msgs.msg import Pose

# Convenience constants.
PI = np.pi

# Threshold near the joint limits at which we consider to be unsafe.
SAFETY_POS_THRESHOLD = 5 * PI / 180  # 5 degrees

SAFETY_VEL_THRESHOLD = 1.0  # 1rad/s

# Current threshold above which we consider as unsafe.
SAFETY_CURRENT_THRESHOLD = 200  # mA

# Mapping of motor ID to (scale, offset).
DEFAULT_DHAND_CALIBRATION_MAP = CalibrationMap({
    # Finger 1
    10: (1, -PI),
    11: (1, -3 * PI / 2),
    12: (1, -PI),
    13: (1, -PI),
    # Finger 2
    20: (1, -PI),
    21: (1, -3 * PI / 2),
    22: (1, -PI),
    23: (1, -PI),
    # Finger 3
    30: (1, -PI),
    31: (1, -3 * PI / 2),
    32: (1, -PI),
    33: (1, -PI),
    # Thumb
    40: (1, -220 * PI / 180),
    41: (1, -PI),
    42: (1, -PI),
    43: (1, -PI),
    # Object
    #50: (1, -PI),
    # Guide
    #60: (1, -PI),
})

class SawyerListener():
    # Receiver node
    def __init__(self):
        import rospy
        from std_msgs.msg import String
        from geometry_msgs.msg import Pose

        self.sawyer_qp = [0,0,0]
        self.sawyer_qpa = [0,0,0,0]
        self.obj_qp = [0,0,0]
        self.dat = np.concatenate((self.sawyer_qp, self.sawyer_qpa, self.obj_qp))
        rospy.Subscriber("get_angles", Pose, self.store_latest_qp)
        rospy.Subscriber("get_obj", Floats, self.store_latest_obj)
        rospy.sleep(1)

    def store_latest_qp(self, pose):
        self.sawyer_qp = [pose.position.x, pose.position.y, pose.position.z]
        self.sawyer_qpa = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        self.update_dat('sawyer')

    def store_latest_obj(self, obj_qp):
        #print(obj_qp)
        self.obj_qp = obj_qp.data[-3:]
        self.update_dat('obj')

    def update_dat(self, which):
        if which=='obj':
            self.dat[-3:] = self.obj_qp
        else:
            self.dat[:3] = self.sawyer_qp
            self.dat[3:-3] = self.sawyer_qpa   
        #dat [Sqpx,Sqpy,Sqpz,qpa,qpa,qpa,qpa,Oqpx,Oqpy,Oqpz] 

class SawyerCommander():
    # Talking node
    def __init__(self):
        import rospy
        from std_msgs.msg import String
        from geometry_msgs.msg import Pose

        rate = rospy.Rate(13) # 10hz
        self.pub = rospy.Publisher("set_angles", Pose, queue_size=10)
        rospy.sleep(1)

    def send_command(self, comm):
        if self.pub.get_num_connections() == 0:
                print("No subscribers connected")
        self.pub.publish(comm)


class BaseDHandEnv(RobotEnv, metaclass=abc.ABCMeta):
    """Base environment for all DHand robot tasks."""

    def __init__(self,
                 *args,
                 device_path: Optional[str] = '/dev/ttyUSB0',
                 sim_observation_noise: Optional[float] = None,
                 frame_skip = 125,
                 **kwargs):
        """Initializes the environment.

        Args:
            device_path: The device path to Dynamixel hardware.
            sim_observation_noise: If given, configures the RobotComponent to
                add noise to observations.
        """
        super().__init__(*args, **kwargs)
        self._device_path = device_path
        self._sim_observation_noise = sim_observation_noise

        # Create the robot component.
        robot_builder = RobotComponentBuilder()
        self._configure_robot(robot_builder)
        self.robot = self._add_component(robot_builder)

	# Create communicators with Sawyer
        import rospy
        rospy.init_node('sawyer2')
        self.listener = SawyerListener()
        self.commander = SawyerCommander()
        self.sawyer_bounds = [[0.4, 0.7],
                       [-0.2, 0.2],
                       [0.2, 0.5],
                       [0,1],
                       [0,1],
                       [0,1],
                       [0,1],]
        
        
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Returns the current state of the environment."""
        state = self.robot.get_state('dhand')
        sawyer_state = self.robot.get_state('sawyer')
        return {'qpos': np.concatenate((sawyer_state, state.qpos)), 'qvel': np.concatenate((np.zeros(7), state.qvel))}

    def set_state(self, state: Dict[str, np.ndarray]):
        """Sets the state of the environment."""
        self.robot.set_state(
            {'sawyer': RobotState(qpos=state['qpos'][:7], qvel=state['qvel'][:7])})
        self.robot.set_state(
            {'dhand': RobotState(qpos=state['qpos'][7:], qvel=state['qvel'][7:])})
    
    def command_sawyer(self, state: Dict[str, np.ndarray]):
        action = state.qpos
        action_range = 2 # -1 to 1
        qp_range = self.robot.get_config('sawyer').qpos_range
        print("ACTION 0: {}".format(action[:3]))
        action = [((action[i] + 1)/action_range*(qp_range[i,1]-qp_range[i,0]))+qp_range[i,0] for i in range(action.size)]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = action[:3]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = action[3:]
        print(pose)
        self.commander.send_command(pose)

    def _configure_robot(self, builder: RobotComponentBuilder):
        """Configures the robot component."""
        # Add the main D'Claw group.
        builder.add_group(
            'dhand',
            qpos_indices=range(16),
            qpos_range=[
                (-0.1, 0.1),
                (-0.61*PI, 0.027*PI),
                (-PI / 2, PI / 2),
                (-PI / 2, 0)
            ] * 4,
            qvel_range=None, #[(-2 * PI / 3, 2 * PI / 3)] * 16,
            actuator_indices=None
            )
        builder.add_group(
            'sawyer',
            qpos_indices=range(7),
            qpos_range=[
                (0.4,0.5),
                (-0.16,0.05),
                (0.3,0.6)]+
                [(0,1)]*4
            ,
            qvel_range=None,
            actuator_indices=None
            )
        if self._sim_observation_noise is not None:
            builder.update_group(
                'dhand', sim_observation_noise=self._sim_observation_noise)
        # If a device path is given, set the motor IDs and calibration map.
        if self._device_path:
            builder.set_dynamixel_device_path(self._device_path)
            builder.set_hardware_calibration_map(DEFAULT_DHAND_CALIBRATION_MAP)
            builder.update_group(
                'dhand', motor_ids=[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43])
            scripted_reset.add_groups_for_reset(builder)

    def _initialize_action_space(self) -> gym.Space:
        """Returns the observation space to use for this environment."""
        qpos_indices = self.robot.get_config('dhand').qpos_indices
        s_qpos_indices = self.robot.get_config('sawyer').qpos_indices
        return make_box_space(-1.0, 1.0, shape=(qpos_indices.size+s_qpos_indices.size,))

    def _get_safety_scores(
            self,
            pos: Optional[np.ndarray] = None,
            vel: Optional[np.ndarray] = None,
            current: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Computes safety-related scores for D'Claw robots.

        Args:
            pos: The joint positions.
            vel: The joint velocities.
            current: The motor currents.

        Returns:
            A dictionary of safety scores for the given values.
        """
        scores = collections.OrderedDict()
        dclaw_config = self.robot.get_config('dhand')

        if pos is not None and dclaw_config.qpos_range is not None:
            # Calculate lower and upper separately so broadcasting works when
            # positions are batched.
            near_lower_limit = (
                np.abs(dclaw_config.qpos_range[:, 0] - pos) <
                SAFETY_POS_THRESHOLD)
            near_upper_limit = (
                np.abs(dclaw_config.qpos_range[:, 1] - pos) <
                SAFETY_POS_THRESHOLD)
            near_pos_limit = np.sum(near_lower_limit | near_upper_limit, axis=1)
            scores['safety_pos_violation'] = near_pos_limit

        if vel is not None:
            above_vel_limit = np.sum(np.abs(vel) > SAFETY_VEL_THRESHOLD, axis=1)
            scores['safety_vel_violation'] = above_vel_limit

        if current is not None:
            above_current_limit = np.sum(
                np.abs(current) > SAFETY_CURRENT_THRESHOLD, axis=1)
            scores['safety_current_violation'] = above_current_limit
        return scores


class BaseDHandObjectEnv(BaseDHandEnv, metaclass=abc.ABCMeta):
    """Base environment for all DClaw robot tasks with objects."""

    def __init__(self, *args, use_guide: bool = False, **kwargs):
        """Initializes the environment.

        Args:
            use_guide: If True, activates an object motor in hardware to use
                to show the goal.
        """
        self._use_guide = use_guide
        super().__init__(*args, **kwargs)

    def get_state(self) -> Dict[str, np.ndarray]:
        """Returns the current state of the environment."""
        self.robot.set_state({
            'object': RobotState(qpos=self.listener.obj_qp, qvel=np.zeros(3)),
            'sawyer': RobotState(qpos=self.listener.dat[:7], qvel=np.zeros(7))})
        print(self.listener.obj_qp)        
        dhand_state, object_state, sawyer_state = self.robot.get_state(['dhand', 'object', 'sawyer'])
        return {
            'dhand_qpos': dhand_state.qpos,
            'dhand_qvel': dhand_state.qvel,
            'object_qpos': object_state.qpos,
            'object_qvel': object_state.qvel,
            'sawyer_qpos': sawyer_state.qpos,
            'sawyer_qvel': sawyer_state.qvel,
        }

    def set_state(self, state: Dict[str, np.ndarray]):
        """Sets the state of the environment."""
        self.robot.set_state({
            'dclaw': RobotState(
                qpos=state['dhand_qpos'], qvel=state['dhand_qvel']),
            'object': RobotState(
                qpos=state['object_qpos'], qvel=state['object_qvel']),
            'sawyer': RobotState(
                qpos=state['sawyer_qpos'], qvel=state['sawyer_qvel']),
        })

    def _configure_robot(self, builder: RobotComponentBuilder):
        """Configures the robot component."""
        super()._configure_robot(builder)
        # Add the object group.
        builder.add_group(
            'object',
            qpos_indices=range(3),  # The object is the last qpos.
            qpos_range=[(0.3,0.6),
                        (-0.2,0.2),
                        (0,0.6)]
        )
        if self._sim_observation_noise is not None:
            builder.update_group(
                'object', sim_observation_noise=self._sim_observation_noise)
        #if self._device_path:
        #    builder.update_group('object', motor_ids=[50])

        # Add the guide group, which is a no-op if the guide motor is unused.
        builder.add_group('guide')
        if self._use_guide and self._device_path:
            builder.update_group('guide', motor_ids=[60], use_raw_actions=True)

    def _reset_dclaw_and_object(
            self,
            dhand_pos: Optional[Sequence[float]] = None,
            dhand_vel: Optional[Sequence[float]] = None,
            object_pos: Optional[Union[float, Sequence[float]]] = None,
            object_vel: Optional[Union[float, Sequence[float]]] = np.zeros(3),
            guide_pos: Optional[Union[float, Sequence[float]]] = None,
            sawyer_pos: Optional[Sequence[float]] = None):

        object_pos = self.listener.obj_qp
        sawyer_pos = self.listener.dat[:7]

        """Reset procedure for DClaw robots that manipulate objects.

        Args:
            claw_pos: The joint positions for the claw (radians).
            claw_vel: The joint velocities for the claw (radians/second). This
                is ignored on hardware.
            object_pos: The joint position for the object (radians).
            object_vel: The joint velocity for the object (radians/second). This
                is ignored on hardware.
            guide_pos: The joint position for the guide motor (radians). The
                guide motor is optional for marking the desired position.
        """
        # Set defaults if parameters are not given.
        dhand_init_state, object_init_state, sawyer_init_state = self.robot.get_initial_state(
            ['dhand', 'object', 'sawyer'])
        dhand_pos = (
            dhand_init_state.qpos if dhand_pos is None else np.asarray(dhand_pos))
        dhand_vel = (
            dhand_init_state.qvel if dhand_vel is None else np.asarray(dhand_vel))
        object_pos = (
            object_init_state.qpos
            if object_pos is None else np.atleast_1d(object_pos))
        object_vel = (
            object_init_state.qvel
            if object_vel is None else np.atleast_1d(object_vel))
        sawyer_pos = (
            sawyer_init_state.qpos
            if sawyer_pos is None else np.asarray(sawyer_pos))
        sawyer_vel = np.zeros(7)
        #guide_pos = (
        #    np.zeros(1) if guide_pos is None else np.atleast_1d(guide_pos))

        if self.robot.is_hardware:
            scripted_reset.reset_to_states(
                self.robot, {
                    'dhand': RobotState(qpos=dhand_pos),
                    'object': RobotState(qpos=object_pos),
                    #'guide': RobotState(qpos=guide_pos),
                })
        else:
            self.robot.set_state({
                'dhand': RobotState(qpos=dhand_pos, qvel=dhand_vel),
                'sawyer': RobotState(qpos=sawyer_pos, qvel=sawyer_vel),
                'object': RobotState(qpos=object_pos, qvel=object_vel),
            })


