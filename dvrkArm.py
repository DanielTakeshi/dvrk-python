import threading
import rospy

import utils as U
import numpy as np
import PyKDL

from tf_conversions import posemath
from std_msgs.msg import String, Bool, Float32, Empty, Float64MultiArray
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState

class dvrkArm(object):
    """Simple arm API wrapping around ROS messages
    """
    def __init__(self, arm_name, ros_namespace='/dvrk'):

        # data members, event based
        self.__arm_name = arm_name
        self.__ros_namespace = ros_namespace
        self.__goal_reached = False
        self.__goal_reached_event = threading.Event()
        self.__get_position = False
        self.__get_position_event = threading.Event()
        self.__get_jaw = False
        self.__get_jaw_event = threading.Event()

        # continuous publish from dvrk_bridge
        self.__position_cartesian_current = PyKDL.Frame()
        self.__position_joint_current = np.array(0, dtype = np.float)
        self.__position_jaw_current = 0.0

        self.__sub_list = []
        self.__pub_list = []

        # publisher
        frame = PyKDL.Frame()
        self.__full_ros_namespace = self.__ros_namespace + self.__arm_name
        self.__set_position_joint_pub = rospy.Publisher(self.__full_ros_namespace + '/set_position_joint', JointState,
                                                        latch = True, queue_size = 1)
        self.__set_position_goal_joint_pub = rospy.Publisher(self.__full_ros_namespace + '/set_position_goal_joint',
                                                             JointState, latch = True, queue_size = 1)
        self.__set_position_cartesian_pub = rospy.Publisher(self.__full_ros_namespace
                                                            + '/set_position_cartesian',
                                                            Pose, latch = True, queue_size = 1)
        self.__set_position_goal_cartesian_pub = rospy.Publisher(self.__full_ros_namespace
                                                                 + '/set_position_goal_cartesian',
                                                                 Pose, latch = True, queue_size = 1)
        self.__set_position_jaw_pub = rospy.Publisher(self.__full_ros_namespace
                                                      + '/set_position_jaw',
                                                      JointState, latch = True, queue_size = 1)
        self.__set_position_goal_jaw_pub = rospy.Publisher(self.__full_ros_namespace
                                                           + '/set_position_goal_jaw',
                                                           JointState, latch = True, queue_size = 1)

        self.__pub_list = [self.__set_position_joint_pub,
                           self.__set_position_goal_joint_pub,
                           self.__set_position_cartesian_pub,
                           self.__set_position_goal_cartesian_pub,
                           self.__set_position_jaw_pub,
                           self.__set_position_goal_jaw_pub]


        self.__sub_list = [rospy.Subscriber(self.__full_ros_namespace + '/goal_reached',
                                          Bool, self.__goal_reached_cb),
                           rospy.Subscriber(self.__full_ros_namespace + '/position_cartesian_current',
                                          PoseStamped, self.__position_cartesian_current_cb),
                           rospy.Subscriber(self.__full_ros_namespace + '/io/joint_position',
                                            JointState, self.__position_joint_current_cb),
                           rospy.Subscriber(self.__full_ros_namespace + '/state_jaw_current',
                                            JointState, self.__position_jaw_current_cb)]

        # create node
        if not rospy.get_node_uri():
            rospy.init_node('dvrkArm_node', anonymous = True, log_level = rospy.WARN)
            self.interval_ms = 30  # Sept 6: Minho suggests 20ms --> 30ms?
            self.rate = rospy.Rate(1000.0 / self.interval_ms)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')

    """
    Callback function
    """
    def __goal_reached_cb(self, data):
        """Callback for the goal reached.
        """
        self.__goal_reached = data.data
        self.__goal_reached_event.set()

    def __position_cartesian_current_cb(self, data):
        """Callback for the current cartesian position.
        """
        self.__position_cartesian_current = posemath.fromMsg(data.pose)
        # self.__get_position = True
        self.__get_position_event.set()

    def __position_joint_current_cb(self, data):
        """Callback for the current joint position.
        """
        self.__position_joint_current.resize(len(data.position))
        self.__position_joint_current.flat[:] = data.position

    def __position_jaw_current_cb(self, data):
        """Callback for the current jaw position.
        """
        self.__position_jaw_current = data.position
        self.__get_jaw = True
        self.__get_jaw_event.set()

    """
    Get States function
    """
    def get_current_pose_frame(self):
        """

        :return: PyKDL.Frame
        """
        return self.__position_cartesian_current

    def get_current_pose(self,unit='rad'):    # Unit: pos in (m) rot in (rad) or (deg)
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.array
        """
        pos,rot = self.PyKDLFrame_to_NumpyArray(self.__position_cartesian_current)
        if unit == 'deg':
            rot = U.rad_to_deg(rot)
        return pos,rot

    def get_current_pose_and_wait(self, unit='rad'):    # Unit: pos in (m) rot in (rad) or (deg)
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.array
        """
        self.__get_position_event.clear()

        # the position is originally not received
        # self.__get_position = False
        # recursively call this function until the position is received
        # self.__get_position_event.wait(20)  # 1 minute at most

        if self.__get_position_event.wait(20):  # 1 minute at most
            pos, rot = self.PyKDLFrame_to_NumpyArray(self.__position_cartesian_current)
            if unit == 'deg':
                rot = U.rad_to_deg(rot)
            return pos, rot
        else:
            return [], []

    def get_current_joint(self, unit='rad'):
        """

        :param unit: 'rad' or 'deg'
        :return: List
        """
        joint = self.__position_joint_current
        if unit == 'deg':
            joint = U.rad_to_deg(self.__position_joint_current)
            joint[2] = self.__position_joint_current[2]
        return joint

    def get_current_jaw(self,unit='rad'):
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.float64
        """
        jaw = np.float64(self.__position_jaw_current)
        if unit == "deg":
            jaw = U.rad_to_deg(self.__position_jaw_current)
        return jaw

    def get_current_jaw_and_wait(self, unit='rad'):    # Unit: pos in (m) rot in (rad) or (deg)
        """

        :param unit: 'rad' or 'deg'
        :return: Numpy.array
        """
        self.__get_jaw_event.clear()

        # the position is originally not received
        self.__get_jaw = False
        # recursively call this function until the position is received
        self.__get_jaw_event.wait(20)  # 1 minute at most

        if self.__get_jaw:
            jaw = np.float64(self.__position_jaw_current)
            if unit == "deg":
                jaw = U.rad_to_deg(self.__position_jaw_current)
            return jaw
        else:
            return []

    """
    Set States function
    """
    def set_pose_frame(self, frame):
        """

        :param frame: PyKDL.Frame
        """
        msg = posemath.toMsg(frame)
        return self.__set_position_goal_cartesian_publish_and_wait(msg)

    def set_pose_quaternion(self, pos, rot, wait_callback=True):
        """

        :param pos: position array [x,y,z]
        :param rot: orientation array in quaternion [x,y,z,w]
        :param wait_callback: True or False
        """
        # set in position cartesian mode
        frame = self.NumpyArraytoPyKDLFrame_quaternion(pos, rot)
        msg = posemath.toMsg(frame)
        # go to that position by goal
        if wait_callback:
            return self.__set_position_goal_cartesian_publish_and_wait(msg)
        else:
            self.__set_position_goal_cartesian_pub.publish(msg)
            return True

    def set_pose(self, pos, rot, unit='rad', wait_callback=True):
        """

        :param pos_des: position array [x,y,z]
        :param rot_des: rotation array [Z,Y,X euler angle]
        :param unit: 'rad' or 'deg'
        :param wait_callback: True or False
        """
        if unit == 'deg':
            rot = U.deg_to_rad(rot)
        # set in position cartesian mode
        frame = self.NumpyArraytoPyKDLFrame(pos, rot)
        msg = posemath.toMsg(frame)
        # go to that position by goal
        if wait_callback:
            return self.__set_position_goal_cartesian_publish_and_wait(msg)
        else:
            self.__set_position_goal_cartesian_pub.publish(msg)
            return True

    def set_pose_direct(self, pos, rot, unit='rad'):
        """

        :param pos_des: position array [x,y,z]
        :param rot_des: rotation array [Z,Y,X euler angle]
        :param unit: 'rad' or 'deg'
        """
        if unit == 'deg':
            rot = U.deg_to_rad(rot)

        # set in position cartesian mode
        frame = self.NumpyArraytoPyKDLFrame(pos, rot)
        msg = posemath.toMsg(frame)
        # go to that position by goal
        self.__set_position_cartesian_pub.publish(msg)

    # specify intermediate points between q0 & qf using linear interpolation (blocked until goal reached)
    def set_pose_linear(self, pos, rot, unit='rad'):

        [q0,trash] = self.get_current_pose_and_wait()
        qf = pos
        assert len(qf) > 0, qf
        assert len(q0) > 0, q0
        
        if np.allclose(q0,qf):
            return False
        else:
            tf = np.linalg.norm(np.array(qf)-np.array(q0))**0.8 * 10
            v_limit = (np.array(qf)-np.array(q0))/tf
            v = v_limit * 1.5
            # print '\n'
            # print 'q0=', q0
            # print 'qf=', qf
            # print 'norm=', np.linalg.norm(np.array(qf) - np.array(q0))
            # print 'tf=', tf
            # print 'v=',v
            t = 0.0
            while True:
                q = self.LSPB(q0, qf, t, tf, v)
                # print q
                self.set_pose(q, rot, unit, False)
                # self.set_pose_direct(q, rot, unit)
                t += 0.001 * self.interval_ms
                self.rate.sleep()
                if t > tf:
                    break

    def __set_position_goal_cartesian_publish_and_wait(self, msg):
        """

        :param msg: pose
        :returns: returns true if the goal is reached
        """
        self.__goal_reached_event.clear()
        # the goal is originally not reached
        self.__goal_reached = False
        # recursively call this function until end is reached
        self.__set_position_goal_cartesian_pub.publish(msg)
        self.__goal_reached_event.wait(20) # 1 minute at most
        if not self.__goal_reached:
            return False
        return True

    def set_joint(self, joint, unit='rad', wait_callback=True):
        """

        :param joint: joint array [j1, ..., j6]
        :param unit: 'rad', or 'deg'
        :param wait_callback: True or False
        """
        if unit == 'deg':
            joint = U.deg_to_rad(joint)
        msg = JointState()
        msg.position = joint
        if wait_callback:
            return self.__set_position_goal_joint_publish_and_wait(msg)
        else:
            self.__set_position_goal_joint_pub.publish(msg)
            return True

    def __set_position_goal_joint_publish_and_wait(self, msg):
        """

        :param msg: there is only one parameter, msg which tells us what the ending position is
        :returns: whether or not you have successfully moved by goal or not
        """
        self.__goal_reached_event.clear()
        # the goal is originally not reached
        self.__goal_reached = False
        # recursively call this function until end is reached
        self.__set_position_goal_joint_pub.publish(msg)
        self.__goal_reached_event.wait(20) # 1 minute at most
        if not self.__goal_reached:
            return False
        return True

    def set_jaw(self, jaw, unit='rad', wait_callback=True):
        """

        :param jaw: jaw angle
        :param unit: 'rad' or 'deg'
        :param wait_callback: True or False
        """
        if unit == 'deg':
            jaw = U.deg_to_rad(jaw)
        msg = JointState()
        msg.position = [jaw]
        if wait_callback:
            return self.__set_position_goal_jaw_publish_and_wait(msg)
        else:
            self.__set_position_goal_jaw_pub.publish(msg)
            return True

    def set_jaw_direct(self, jaw, unit='rad'):
        """

        :param jaw: jaw angle
        :param unit: 'rad' or 'deg'
        """
        if unit == 'deg':
            jaw = U.deg_to_rad(jaw)
        msg = JointState()
        msg.position = [jaw]
        self.__set_position_jaw_pub.publish(msg)

    # specify intermediate points between q0 & qf using linear interpolation (blocked until goal reached)
    def set_jaw_linear(self, jaw, unit='rad'):

        q0 = self.get_current_jaw_and_wait()
        qf = jaw
        if np.allclose(q0,qf):
            return False
        else:
            tf = np.linalg.norm(np.array(qf)-np.array(q0))**0.8 * 0.6
            v_limit = (np.array(qf)-np.array(q0))/tf
            v = v_limit * 1.5
            t = 0.0
            while True:
                q = self.LSPB(q0, qf, t, tf, v)
                # print q
                # self.set_pose(q, rot, unit, False)
                self.set_jaw_direct(q, unit)
                t += 0.001 * self.interval_ms
                self.rate.sleep()
                if t > tf:
                    break

    def __set_position_goal_jaw_publish_and_wait(self,msg):
        """

        :param msg:
        :return: whether or not you have successfully moved by goal or not
        """
        self.__goal_reached_event.clear()
        # the goal is originally not reached
        self.__goal_reached = False
        # recursively call this function until end is reached
        self.__set_position_goal_jaw_pub.publish(msg)
        self.__goal_reached_event.wait(20) # 1 minute at most
        if not self.__goal_reached:
            return False
        return True

    """
    Conversion function
    """
    def PyKDLFrame_to_NumpyArray(self,frame):
        pos = np.array([frame.p[0], frame.p[1], frame.p[2]])
        rz, ry, rx = self.__position_cartesian_current.M.GetEulerZYX()
        rot = np.array([np.pi/2, 0, np.pi]) - np.array([rz, ry, rx])
        return pos,rot

    def NumpyArraytoPyKDLFrame(self,pos,rot):
        px, py, pz = pos
        rz, ry, rx = np.array([np.pi / 2, 0, -np.pi]) - np.array(rot)
        return PyKDL.Frame(PyKDL.Rotation.EulerZYX(rz, ry, rx), PyKDL.Vector(px, py, pz))

    def NumpyArraytoPyKDLFrame_quaternion(self,pos,rot):
        px, py, pz = pos
        rx, ry, rz, rw = rot
        return PyKDL.Frame(PyKDL.Rotation.Quaternion(rx, ry, rz, rw), PyKDL.Vector(px, py, pz))

    """
    Trajectory
    """
    def LSPB(self, q0, qf, t, tf, v):

        if np.allclose(q0,qf):    return q0
        elif np.all(v)==0:    return q0
        elif tf==0:     return q0
        elif tf<0:     return []
        elif t<0:      return []
        else:
            v_limit = (np.array(qf) - np.array(q0)) / tf
            if np.allclose(U.normalize(v),U.normalize(v_limit)):
                if np.linalg.norm(v) < np.linalg.norm(v_limit) or np.linalg.norm(2*v_limit) < np.linalg.norm(v):
                    return []
                else:
                    tb = np.linalg.norm(np.array(q0)-np.array(qf)+np.array(v)*tf) / np.linalg.norm(v)
                    a = np.array(v)/tb
                    if 0 <= t and t < tb:
                        q = np.array(q0) + np.array(a)/2*t*t
                    elif tb < t and t <= tf - tb:
                        q = (np.array(qf)+np.array(q0)-np.array(v)*tf)/2 + np.array(v)*t
                    elif tf - tb < t and t <= tf:
                        q = np.array(qf)-np.array(a)*tf*tf/2 + np.array(a)*tf*t - np.array(a)/2*t*t
                    else:
                        return []
                    return q
            else:
                return []

if __name__ == "__main__":
    p = dvrkArm('/PSM2')
    # pos_des = [0.1, 0.10, -0.1]  # Position (m)
    pos_des = [0.0, 0.0, -0.14]  # Position (m)
    rot_des = [0, 0, 0]  # Euler angle ZYX (or roll-pitch-yaw)
    # jaw_des = [0]
    # p.set_pose(pos_des, rot_des, 'deg')
    # joint = [0, 0, 0.15, 0, 0, 0]
    # ps.set_joint(joint)
    jaw = 0
    p.set_jaw(jaw, 'deg')
    # p.set_pose_linear(pos_des,rot_des)
    # print p.get_current_pose_frame()
    # print p.get_current_pose('deg')
    # print p.get_current_joint('deg')
    # print p.get_current_jaw('deg')
