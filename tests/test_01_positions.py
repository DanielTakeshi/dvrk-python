"""
Use this script to simply get the pose of the dvrk. Move it to a spot, record
pose, etc. This is to confirm if real code will do the same thing.

python test_01_positions.py

NOTE: if you find that the positions aren't getting assigned correctly (e.g., if
you make the arm go to 0 degrees yaw but it's always stuck at 100 degrees, for
example) then it's likely that either something's broken, or that the arm is
simply near or at the limit of its 'workspace.' For the latter, simply rotate
the arm back to more of a 'central' state, if that makes sense.

NOTE: might as well use this for calibration.
"""
import sys
sys.path.append('..')
from dvrkArm import dvrkArm
from dvrkClothSim import dvrkClothSim
import time
import math
import rospy
import numpy as np
np.set_printoptions(suppress=True)


def _calibrate(p):
    pose_rad = p.get_current_pose()
    pose_deg = p.get_current_pose(unit='deg')
    pos, rot = pose_deg
    #print('starting pose (deg):\n{}\n{}'.format(pos,rot))
    print('starting pose (deg): {}'.format(pose_deg))


if __name__ == "__main__":
    p = dvrkArm('/PSM2')
    _calibrate(p)
    sys.exit()

    # Other stuff.
    pose_frame = p.get_current_pose_frame()
    pose_rad = p.get_current_pose()
    pose_deg = p.get_current_pose(unit='deg')
    jaw = p.get_current_jaw(unit='deg')
    #print('starting jaw: {}'.format(jaw))

    #print('pose_frame:\n{}'.format(pose_frame))
    #print('starting pose (rad): {}'.format(pose_rad))
    print('starting pose (deg): {}'.format(pose_deg))

    # was this deprecated?
    #joint = p.get_current_joint(unit='deg')
    #print('joint: {}'.format(joint))

    # You can set the gripper angle.
    p.set_jaw(jaw=0, unit='deg')

    # We can force it to go to the origin position.
    org_pos = [0.003, 0.001, -0.06]
    org_rot = [0, 0, 0]
    p.set_pose_linear(org_pos, org_rot, 'deg')
    time.sleep(2)
    print('after moving, pose (deg): {}'.format(p.get_current_pose(unit='deg')))

