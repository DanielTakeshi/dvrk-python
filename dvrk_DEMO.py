from dvrkArm import dvrkArm
from dvrkClothsim import  dvrkClothsim
import rospy
import numpy as np
import time


p1 = dvrkArm('/PSM1')
p2 = dvrkArm('/PSM2')
t = 0.0
MILLION = 10**6
interval_ms = 10
r = rospy.Rate(1000.0 / interval_ms)

def move_origin():
    pos_org = [0.0, 0.0, -0.13]
    # Position (m)
    rot_org = [0, 0, 0]  # Euler angle ZYX (or roll-pitch-yaw)
    jaw_org = [0]
    p1.set_pose(pos_org, rot_org, 'deg')
    p1.set_jaw(jaw_org, 'deg')

def make_teleop_conf():
    pos_des1 = [0.06, -0.07, -0.125]
    # rot_des1 = [-88.7, -79.6, 312.2]
    rot_des1 = [-25.3, -72.6, 343.2]
    jaw_des1 = [30]
    p1.set_pose(pos_des1, rot_des1, 'deg')
    p1.set_jaw(jaw_des1, 'deg')

    pos_des2 = [-0.09, -0.07, -0.128]
    # rot_des2 = [219.7, 73.3, 19.2]
    rot_des2 = [17.3, -78.3, 332.2]
    jaw_des2 = [30]
    p2.set_pose(pos_des2, rot_des2, 'deg')
    p2.set_jaw(jaw_des2, 'deg')

def init_conf_curve():
    x_offset = 0.04
    y_offset = 0.04
    pos_curve_org = [x_offset, y_offset, -0.13]
    rot_curve_org = [0.0, 0.0, 0.0]
    jaw_curve_org = [0]
    p1.set_pose(pos_curve_org, rot_curve_org, 'deg')
    p1.set_jaw(jaw_curve_org, 'deg')

    x_offset2 = -0.04
    y_offset2 = 0.04
    pos_curve_org2 = [x_offset2, y_offset2, -0.13]
    rot_curve_org2 = [0.0, 0.0, 0.0]
    jaw_curve_org2 = [0]
    p2.set_pose(pos_curve_org2, rot_curve_org2, 'deg')
    p2.set_jaw(jaw_curve_org2, 'deg')

T = 3   # period
def make_curve(t):
    # Lemniscate of Bernoulli
    a = 0.03
    ratio = 0.8
    t_ = t/T*2*np.pi + np.pi/2
    x = ratio*(a * 2 ** 0.5 * np.cos(t_)) / (np.sin(t_) * np.sin(t_) + 1)
    y = (a * 2 ** 0.5 * np.cos(t_) * np.sin(t_)) / (np.sin(t_) * np.sin(t_) + 1)
    jaw = 25*np.sin(2*np.pi*t) + 25
    x_offset = 0.04
    y_offset = 0.04
    pos_lem = [x+x_offset, y+y_offset, -0.13]
    rot_lem = [0.0, 0.0, 0.0]
    jaw_lem = [jaw]
    p1.set_pose(pos_lem, rot_lem, 'deg', False)
    p1.set_jaw(jaw_lem, 'deg', False)

    x = ratio*(a * 2 ** 0.5 * np.cos(t_)) / (np.sin(t_) * np.sin(t_) + 1)
    y = (a * 2 ** 0.5 * np.cos(t_) * np.sin(t_)) / (np.sin(t_) * np.sin(t_) + 1)
    jaw2 = 25 * np.sin(2 * np.pi * t) + 25
    x_offset2 = -0.04
    y_offset2 = 0.04
    pos_lem = [x+x_offset2, y+y_offset2, -0.13]
    rot_lem = [0.0, 0.0, 0.0]
    jaw_lem = [jaw2]
    p2.set_pose(pos_lem, rot_lem, 'deg', False)
    p2.set_jaw(jaw_lem, 'deg', False)

if __name__ == "__main__":
    # print p1.get_current_pose('deg')
    # time.sleep(1)
    # print p2.get_current_pose('deg')
    DEMO = 'curve'
    if DEMO == 'teleop':
        make_teleop_conf()
    elif DEMO == 'curve':
        init_conf_curve()
        while not rospy.is_shutdown():
            try:
                make_curve(t)
                t += 1000.0 / MILLION * interval_ms
                r.sleep()
            except rospy.ROSInterruptException:
                pass
    elif DEMO == 'cloth':
        p = dvrkClothsim()
        p.set_position_origin([0.0, 0.03, -0.14], 0, 'deg')
        while True:
            p.move_pose_pickup([0.06, 0.0], [0.04, 0.02], 0, 'deg')
            p.move_pose_pickup([0.06, 0.05], [0.04, 0.03], 0, 'deg')
            p.move_pose_pickup([0.0, 0.05], [0.02, 0.03], 0, 'deg')