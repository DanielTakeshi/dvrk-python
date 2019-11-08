from dvrkArm import dvrkArm
import threading
import time
import rospy
import numpy as np
import utils as U


class dvrkClothSim(threading.Thread):
    """
    Motion library for dvrk
    """
    def __init__(self, interval_ms=10.0, ros_namespace='/dvrk'):
        threading.Thread.__init__(self)
        self.__ros_namespace = ros_namespace
        self.interval_ms = interval_ms
        self.arm = dvrkArm('/PSM2')
        self.t = 0.0
        self.nStart = 0.0
        self.nEnd = 0.0
        self.stop_flag = False
        self.rate = rospy.Rate(1000.0 / self.interval_ms)

        # Motion variables
        self.pos_org = [0.0, 0.0, -0.12]  #   xyz position in (m)
        self.rot_org = [0.0, 0.0, 0.0]    #   ZYX Euler angle in (rad)
        self.pos_pick = [0.0, 0.0]        #   xy coordinate for the cloth pick-up
        self.rot_pick = [0.0, 0.0, 0.0]   #   ZYX Euler angle in (rad)
        # self.pick_depth = -0.12
        self.pickup_height = -0.115
        self.jaw_opening = 50*3.141592/180.0
        self.jaw_closing = -5*3.141592/180.0

    """
    Motion Creating function for cloth simulation
    """
    def set_position_origin(self, pos, rot, unit='rad'):
        self.rot_org[0] = rot
        if unit == 'deg':
            self.rot_org[0] = rot*3.141592/180.0
        self.pos_org = pos

    def move_pose_pickup(self, pos_pick, pos_drop, rot_pick, unit='rad',
            only_do_pick=False):
        """The main arm motion we should be using.

        :param pos_pick: x,y coordinate to pick up (in background space)
        :param pos_drop: x,y coordinate to drop (in background spcae)
        :param rot_pick: roll angle of grasper to pick up
        :param unit: position in (m) and rotation in (deg) or (rad)
        :param only_do_pick: Extra debugging layer added by Daniel for
            calibration, this will only go to the target. Use this to check if
            it is going to the right target.
        :return:
        """
        if unit == 'deg':
            rot_pick = rot_pick*3.141591/180.0

        # move to the origin
        self.arm.set_pose_linear(self.pos_org, self.rot_org, 'rad')

        # Daniel: added this.
        rospy.sleep(1)

        # move upon the pick-up spot and open the jaw
        p_temp = np.array([pos_pick[0], pos_pick[1], self.pickup_height])
        r_temp = np.array([rot_pick, 0.0, 0.0])
        self.arm.set_pose_linear(p_temp, r_temp, 'rad')
        self.arm.set_jaw(self.jaw_opening, False)

        # move downward and grasp the cloth
        pos_downward = np.array([pos_pick[0], pos_pick[1], pos_pick[2]])
        self.arm.set_pose_linear(pos_downward, r_temp, 'rad')
        self.arm.set_jaw(self.jaw_closing, False)

        if only_do_pick:
            return

        # move upward, move the cloth, and drop the cloth
        self.arm.set_pose_linear(p_temp, r_temp, 'rad')
        p_temp2 = np.array([pos_drop[0], pos_drop[1], self.pickup_height])
        self.arm.set_pose_linear(p_temp2, r_temp, 'rad')
        self.arm.set_jaw(self.jaw_opening, False)

        # move to the origin and close the jaw
        self.arm.set_pose_linear(self.pos_org, self.rot_org, 'rad')

    def start(self):
        self.stop_flag = False
        self.thread = threading.Thread(target=self.run, args=(lambda: self.stop_flag,))
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.stop_flag = True

    def run(self, stop):
        while True:
            self.nEnd = time.clock() * U.MILLION  # (us)
            if self.nEnd - self.nStart < self.interval_ms * 1000:
                pass
            else:
                # To do here
                self.t += 1000.0 / U.MILLION * self.interval_ms
                self.nStart = self.nEnd;
            if stop():
                break


if __name__ == "__main__":
    p = dvrkClothSim()
    p.set_position_origin([0.003, 0.001, -0.06], 0, 'deg')
    # print(p.arm.get_current_pose())
    # print(p.arm.get_current_pose('deg'))

    #while True:
    #    p.move_pose_pickup([-0.081,0.028,-0.144], [-0.037, 0.018], 0, 'deg')

    #    # p.move_pose_pickup([0.08, 0.07], [0.06, 0.05], 0, 'deg')
    #    # p.move_pose_pickup([0.0, 0.07], [0.02, 0.05], 0, 'deg')
