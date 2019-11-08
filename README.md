# Da Vinci Code

Requires Python 2.7 and using our davinci0 machine.

Use the following core API:

- `dvrkArm.py`: an API wrapping around the ROS messages to get or set positions.
- `dvrkClothSim.py` for moving the arms.

## Experimental Usage

Performing our experiments involves several steps, due to loading a separate
neural network (using a Python3 virtualenv). Roughly, the steps are:

1. Activate the robot via `roscore`, then (in a separate tab) run `./teleop` in
`~/catkin_ws` and click HOME.

2. In another tab in `~/catkin_ws/src/zivid-ros/zivid_samples/launch`, run the
camera capture script:

  ```
  roslaunch zivid_samples sample_capture_py.launch 
  ```

  This will open up rviz, and generate some flashing lights.

3. In yet another tab, *set up a new Python 3 virtualenv*, and run `python
load_net.py`. See `call_network/README.md` for detailed instructions.  This
script runs continuously in the background and checks for any new images in the
target directory.

4. Check the configuration file in `config.py` which will contain a bunch of
settings we can adjust. *In particular, adjust which calibration data we should
be using*. They are in the form of text files.

5. Finally, run `python run.py` for experiments, using the system Python. This
requires some keyboard strokes. *The code runs one episode* and then terminates.
Repeat for more episodes.

Other helpful pointers:

- You can run `python camera.py`, *using the system Python on our machine*
  (i.e., not a virtualenv). The main method shows how to periodically query and
  save images from the camera into `.png` files.  We import this in code above.

- For quick testing of the da vinci with resorting to the machinery of the
  experimental pipeline, use `python dvrkClothSim.py` (for generic motion) or
  one of the test scripts in `tests/` (for calibration accuracy).

- For calibration, we do this to can map from neural network coordinates to
  actions. We use a grid of known points, and then apply bilinear interpolation
  to approximate positions everywhere else.


## Quick Start

Here's a minimal working example. Put this set of code in `tests/test_01_positions.py`:

```python
import sys
sys.path.append('.')
import dvrkArm

if __name__ == "__main__":
    p = dvrkArm.dvrkArm('/PSM1')
    pose_frame = p.get_current_pose_frame()
    pose_rad = p.get_current_pose()
    pose_deg = p.get_current_pose(unit='deg')

    print('pose_frame:\n{}'.format(pose_frame))
    print('pose_rad: {}'.format(pose_rad))
    print('pose_deg: {}'.format(pose_deg))

    targ_pos = [0.04845971,  0.05464517, -0.12231114]
    targ_rot = [4.65831872, -0.69974499,  0.87412989]
    p.set_pose(pos=targ_pos, rot=targ_rot, unit='rad')
```

and run `python tests/test_01_positions.py`. This will print out pose
information about the dvrk arm, which can then be used to hard code some
actions. For example, above we show how to set the pose of the arm given
pre-computed `targ_pos` and `targ_rot`.


