"""Test if calibration is working. This can be run as a stand alone file, i.e.:

python test_03_checkerboard_pick_test.py

should be enough. Be careful that the position is safe!  You can do a sys.exit
and just do one action repeatedly for each corner of the checkerboard. CHECK
WHICH MAPPING GRID WE'RE USING!
"""
import sys
sys.path.append('..')
import numpy as np
from dvrkClothSim import dvrkClothSim
np.set_printoptions(suppress=True)
import utils as U
import time

if __name__ == "__main__":
    row_board = 6
    column_board = 6
    cloth_height = 0.0    # unit = (m)
    data_square = U.load_mapping_table(row_board,
                                       column_board,
                                       'mapping_table_09-04_second_calib',
                                       cloth_height)
    p = dvrkClothSim()
    p.set_position_origin([0.003, 0.001, -0.06], 0, 'deg')
    pose_deg = p.arm.get_current_pose(unit='deg')
    print('current arm pose: {}'.format(pose_deg))

    # Do this just to do one action, to check if calibration is working. Make
    # dx,dy both zero so the robot does not move to a second spot.
    x = 1.0
    y = 1.0
    dx = 0.0
    dy = 0.0
    U.move_p_from_net_output(x, y, dx, dy, row_board, column_board, data_square,
            p, debug=True, only_do_pick=True)
    sys.exit()

    # Do this to go through all points.
    for i in range(row_board):
        for j in range(column_board):
            x = -1 + j * 0.4
            y = -1 + i * 0.4
            dx = 0.0
            dy = 0.0
            print('\n({},{})'.format(x,y))
            U.move_p_from_net_output(x, y, dx, dy, row_board, column_board,
                                     data_square, p, debug=True)
            time.sleep(1)
