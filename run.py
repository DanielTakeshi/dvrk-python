"""Use this for the main experiments. It runs one episode only.

NOTE: I am running into this error with a line in the camera file:

Traceback (most recent call last):
  File "run.py", line 73, in <module>
    cam = camera.RGBD()
  File "/home/davinci0/seita/dvrk_python/camera.py", line 18, in __init__
    rospy.init_node("camera")
  File "/opt/ros/kinetic/lib/python2.7/dist-packages/rospy/client.py", line 274, in init_node
    raise rospy.exceptions.ROSException("rospy.init_node() has already been called with different arguments: "+str(_init_node_args))
rospy.exceptions.ROSException: rospy.init_node() has already been called with different arguments: ('dvrkArm_node', ['run.py'], True, 4, False, False

This code (`run.py`) will run fine if we just comment that line out, but I worry
if that functionality is needed.
"""
import argparse
import os
import sys
import time
import datetime
import cv2
import pickle
import logging
import numpy as np
np.set_printoptions(suppress=True)
from collections import defaultdict
from os.path import join
from skimage.measure import compare_ssim
# Stuff from our code base.
import utils as U
import config as C
from dvrkClothSim import dvrkClothSim
import camera


def _process_images(c_img, d_img, args, debug=True):
    """Process images to make it suitable for deep neural networks.
    
    Mostly mirrors my tests in `camera.py`.
    """
    assert d_img.shape == (1200, 1920), d_img.shape
    assert c_img.shape == (1200, 1920, 3), c_img.shape

    if debug:
        nb_items = np.prod(np.shape(c_img))
        nb_not_nan = np.count_nonzero(~np.isnan(c_img))
        perc = nb_not_nan/float(nb_items)*100
        print('RGB image shape {}, has {} items'.format(c_img.shape, nb_items))
        print('  num NOT nan: {}, or {:.2f}%'.format(nb_not_nan, perc))
        nb_items = np.prod(np.shape(d_img))
        nb_not_nan = np.count_nonzero(~np.isnan(d_img))
        perc = nb_not_nan/float(nb_items)*100
        print('depth image shape {}, has {} items'.format(d_img.shape, nb_items))
        print('  num NOT nan: {}, or {:.2f}%'.format(nb_not_nan, perc))
    c_img[np.isnan(c_img)] = 0
    d_img[np.isnan(d_img)] = 0

    # We `inpaint` to fill in the zero pixels, done on raw depth values. Skip if
    # we're not doing color images due to time? Though we have to be careful if
    # we want to report both color/depth together? Eh just do together ...
    if C.IN_PAINT:
        d_img = U.inpaint_depth_image(d_img, ix=100, iy=500)

    # Process image, but this really means cropping!
    c_img_crop = camera.process_img_for_net(c_img, ix=0, iy=0)
    d_img_crop = camera.process_img_for_net(d_img, ix=100, iy=500)
    assert c_img_crop.shape[0] == c_img_crop.shape[1], c_img.shape

    # Check depth values (in meters), only for the CROPPED regions!
    if debug:
        print('\nAfter NaN filtering, ... for CROPPED depth image:')
        print('  max: {:.3f}'.format(np.max(d_img_crop)))
        print('  min: {:.3f}'.format(np.min(d_img_crop)))
        print('  mean: {:.3f}'.format(np.mean(d_img_crop)))
        print('  medi: {:.3f}'.format(np.median(d_img_crop)))
        print('  std: {:.3f}'.format(np.std(d_img_crop)))
        print('also, types for color/depth: {},{}'.format(
                c_img_crop.dtype, d_img_crop.dtype))
        print('')

    # Let's process depth, from the cropped one, b/c we don't want values
    # out the cropped region to influence any depth 'scaling' calculations.
    d_img_crop = camera.depth_to_3ch(d_img_crop,
                                     cutoff_min=C.CUTOFF_MIN,
                                     cutoff_max=C.CUTOFF_MAX)
    d_img_crop = camera.depth_3ch_to_255(d_img_crop)

    # Try blurring depth, bilateral recommends 9 for offline applications
    # that need heavy blurring. The two sigmas were 75 by default.
    d_img_crop_blur = cv2.bilateralFilter(d_img_crop, 9, 100, 100)
    #d_img_crop_blur = cv2.medianBlur(d_img_crop_blur, 5)

    # TODO: adjust mean pixel values and/or brightness? Not currently doing.
    #c_img = U._adjust_gamma(c_img, gamma = 1.4)

    # Sept 7: Actually try de-noising? That might help a lot!!
    print('Now de-noising (note: color/depth are types {}, {})'.format(
            c_img_crop.dtype, d_img_crop.dtype))  # d_img is a float
    c_img_crop = cv2.fastNlMeansDenoisingColored(c_img_crop, None, 7, 7, 7, 21)
    d_img_crop = cv2.fastNlMeansDenoising(d_img_crop, None, 7, 7, 21)

    # Let's save externally but we can do quick debugging here.
    U.save_image_numbers('tmp', img=c_img_crop, indicator='c_img', debug=True)
    U.save_image_numbers('tmp', img=d_img_crop, indicator='d_img', debug=True)

    return c_img_crop, d_img_crop


def run(args, cam, p):
    """Run one episode, record statistics, etc."""
    stats = defaultdict(list)
    COVERAGE_SUCCESS = 0.92
    exponent = 0

    for i in range(args.max_ep_length):
        print('\n*************************************')
        print('ON TIME STEP (I.E., ACTION) NUMBER {}'.format(i+1))
        print('*************************************\n')

        # ----------------------------------------------------------------------
        # STEP 1: query the image from the camera class using `cam`. To avoid
        # the flashing strobe light, you have to move to the tab with the camera.
        # ----------------------------------------------------------------------
        c_img_raw = None
        d_img_raw = None
        print('Waiting for c_img, & d_img; please press ENTER in the appropriate tab')
        while c_img_raw is None:
            c_img_raw = cam.read_color_data()
        while d_img_raw is None:
            d_img_raw = cam.read_depth_data()
        print('  obtained the (raw) c_img and d_img')

        # ----------------------------------------------------------------------
        # STEP 2: process image and save as a 100x100 png, see `camera.py` for some
        # tests. Image must be saved in specified DVRK_IMG_PATH for the net to see.
        # Also, if coverage is high enough, EXIT NOW!
        # ----------------------------------------------------------------------
        c_img, d_img = _process_images(c_img_raw, d_img_raw, args)
        assert c_img.shape == (100,100,3), c_img.shape
        assert d_img.shape == (100,100,3), d_img.shape
        if args.use_color:
            c_tail = "c_img_{}.png".format(str(i).zfill(2))
            img_path = join(C.DVRK_IMG_PATH, c_tail)
            cv2.imwrite(img_path, c_img)
        else:
            d_tail = "d_img_{}.png".format(str(i).zfill(2))
            img_path = join(C.DVRK_IMG_PATH, d_tail)
            cv2.imwrite(img_path, d_img)
        print('just saved to: {}\n'.format(img_path))
        U.single_means(c_img, depth=False)
        U.single_means(d_img, depth=True)

        coverage = U.calculate_coverage(c_img)

        # Ensures we save the final image in case we exit and get high coverage.
        # Make sure it happens BEFORE the `break` command below so we get final imgs.
        stats['coverage'].append(coverage)
        stats['c_img'].append(c_img)
        stats['d_img'].append(d_img)

        if coverage > COVERAGE_SUCCESS:
            print('\nCOVERAGE SUCCESS: {:.3f} > {:.3f}, exiting ...\n'.format(
                    coverage, COVERAGE_SUCCESS))
            break
        else:
            print('\ncurrent coverage: {:.3f}\n'.format(coverage))
        print('  now wait a few seconds for network to run')
        time.sleep(5)

        # ----------------------------------------------------------------------
        # STEP 3: get the output from the neural network loading class (you did
        # run it in a separate terminal tab, right?) and then show it to a human.
        # HUGE ASSUMPTION: that the last text file indicates the action we want.
        # ----------------------------------------------------------------------
        dvrk_action_paths = sorted(
                [join(C.DVRK_IMG_PATH,x) for x in os.listdir(C.DVRK_IMG_PATH) \
                    if x[-4:]=='.txt']
        )
        assert len(dvrk_action_paths) > 0, 'Did you run the neural net code??'
        action = np.loadtxt(dvrk_action_paths[-1])
        print('neural net says: {}'.format(action))
        stats['actions'].append(action)

        # ----------------------------------------------------------------------
        # STEP 3.5, only if we're not on the first action, if current image is
        # too similar to the old one, move the target points closer towards the
        # center of the cloth plane. An approximation but likely 'good enough'.
        # It does assume the net would predict a similiar action, though ...
        # ----------------------------------------------------------------------
        if i > 0:
            # AH! Go to -2 because I modified code to append (c_img,d_img) above.
            prev_c = stats['c_img'][-2]
            prev_d = stats['d_img'][-2]
            diff_l2_c = np.linalg.norm(c_img - prev_c) / np.prod(c_img.shape)
            diff_l2_d = np.linalg.norm(d_img - prev_d) / np.prod(d_img.shape)
            diff_ss_c = compare_ssim(c_img, prev_c, multichannel=True)
            diff_ss_d = compare_ssim(d_img[:,:,0], prev_d[:,:,0])
            print('\n  (c) diff L2: {:.3f}'.format(diff_l2_c))
            print('  (d) diff L2: {:.3f}'.format(diff_l2_d))
            print('  (c) diff SS: {:.3f}'.format(diff_ss_c))
            print('  (d) diff SS: {:.3f}\n'.format(diff_ss_d))
            stats['diff_l2_c'].append(diff_l2_c)
            stats['diff_l2_d'].append(diff_l2_d)
            stats['diff_ss_c'].append(diff_ss_c)
            stats['diff_ss_d'].append(diff_ss_d)

            # Apply action 'compression'? A 0.95 cutoff empirically works well.
            ss_thresh = 0.95
            if diff_ss_c > ss_thresh:
                exponent += 1
                print('NOTE structural similiarity exceeds {}'.format(ss_thresh))
                action[0] = action[0] * (0.9 ** exponent)
                action[1] = action[1] * (0.9 ** exponent)
                print('revised action after \'compression\': {} w/exponent {}'.format(
                        action, exponent))
            else:
                exponent = 0
 
        # ----------------------------------------------------------------------
        # STEP 4. If the output would result in a dangerous position, human
        # stops by hitting ESC key. Otherwise, press any other key to continue.
        # The human should NOT normally be using this !!
        # ----------------------------------------------------------------------
        title = '{} -- ESC TO CANCEL (Or if episode done)'.format(action)
        if args.use_color:
            exit = U.call_wait_key( cv2.imshow(title, c_img) )
        else:
            exit = U.call_wait_key( cv2.imshow(title, d_img) )
        cv2.destroyAllWindows()
        if exit:
            print('Warning: why are we exiting here?')
            print('It should exit naturally due to (a) coverage or (b) time limits.')
            break

        # ----------------------------------------------------------------------
        # STEP 5: Watch the robot do its action. Terminate the script if the
        # resulting action makes things fail spectacularly.
        # ----------------------------------------------------------------------
        x  = action[0]
        y  = action[1]
        dx = action[2]
        dy = action[3]
        U.move_p_from_net_output(x, y, dx, dy,
                                 row_board=C.ROW_BOARD,
                                 col_board=C.COL_BOARD,
                                 data_square=C.DATA_SQUARE,
                                 p=p)

        # ----------------------------------------------------------------------
        # STEP 6. Record statistics. Sleep just in case, also reset images.
        # Don't save raw images -- causes file sizes to blow up.
        # ----------------------------------------------------------------------
        cam.set_color_none()
        cam.set_depth_none()
        print('Reset color/depth in camera class, waiting a few seconds ...')
        time.sleep(3)

    # If we ended up using all actions above, we really need one more image.
    if len(stats['c_img']) == args.max_ep_length:
        assert len(stats['coverage']) == args.max_ep_length, len(stats['coverage'])
        c_img_raw = None
        d_img_raw = None
        print('Waiting for FINAL c_img, & d_img; please press ENTER in the appropriate tab')
        while c_img_raw is None:
            c_img_raw = cam.read_color_data()
        while d_img_raw is None:
            d_img_raw = cam.read_depth_data()
        c_img, d_img = _process_images(c_img_raw, d_img_raw, args)
        coverage = U.calculate_coverage(c_img)
        stats['coverage'].append(coverage)
        stats['c_img'].append(c_img)
        stats['d_img'].append(d_img)
        print('(for full length episode) final coverage: {:.3f}'.format(coverage))

    # Final book-keeping and return statistics.
    print('\nEPISODE DONE!')
    print('  coverage: {}'.format(stats['coverage']))
    print('  len(coverage): {}'.format(len(stats['coverage'])))
    print('  len(c_img): {}'.format(len(stats['c_img'])))
    print('  len(d_img): {}'.format(len(stats['d_img'])))
    print('  len(actions): {}'.format(len(stats['actions'])))

    # File path shenanigans.
    if args.use_color:
        if args.use_other_color:
            save_path = join('results', 'tier{}_color_yellowcloth'.format(args.tier))
        else:
            save_path = join('results', 'tier{}_color'.format(args.tier))
    else:
        if args.use_other_color:
            save_path = join('results', 'tier{}_depth_yellowcloth'.format(args.tier))
        else:
            save_path = join('results', 'tier{}_depth'.format(args.tier))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    count = len([x for x in os.listdir(save_path) if 'ep_' in x and '.pkl' in x])
    save_path = join(
            save_path,
            'ep_{}_{}.pkl'.format(str(count).zfill(3), U.get_date())
    )
    print('All done with episode! Saving stats to: {}'.format(save_path))
    with open(save_path, 'wb') as fh:
        pickle.dump(stats, fh)
    return stats


if __name__ == "__main__":
    # I would just set all to reasonable defaults, or put them in the config file.
    parser= argparse.ArgumentParser()
    parser.add_argument('--use_other_color', action='store_true')
    parser.add_argument('--use_color', type=int) # 1 = True
    parser.add_argument('--tier', type=int)
    parser.add_argument('--max_ep_length', type=int, default=10)
    args = parser.parse_args()
    assert args.tier is not None
    assert args.use_color is not None
    print('Running with arguments:\n{}'.format(args))

    # Setup
    p = dvrkClothSim()
    p.set_position_origin([0.003, 0.001, -0.060], 0, 'deg')
    cam = camera.RGBD()

    assert os.path.exists(C.CALIB_FILE), C.CALIB_FILE

    # Run one episode.
    stats = run(args, cam, p)

