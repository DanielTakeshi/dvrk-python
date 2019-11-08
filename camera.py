#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse, cv2, math, os, rospy, sys, threading, time
from pprint import pprint
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, JointState
from cv_bridge import CvBridge, CvBridgeError
from os.path import join
#import tf
#import tf2_ros
#import tf2_geometry_msgs
#import IPython
import utils as U


class RGBD(object):

    def __init__(self, init_camera=False):
        """
        Daniel: unfortunately if we leave rospy.init_node('camera') on, and we
        are importing this class in other code, then we're going to run into
        ROS problems. But we need this if we are calling RGBD as a stand alone
        class, such as in the main method of this file at the bottom. I'd just
        leave init_camera as False by default.
        """
        if init_camera:
            rospy.init_node("camera")
        # rostopic list [-s for subscribers] [-p for publishers] [-v verbose]
        self.bridge = CvBridge()
        self.img_rgb_raw = None
        self.img_depth_raw = None
        self.info = None
        self.is_updated = False

        # rospy.Subscriber(name, data_msg_class, callback)
        # Use `rqt_image_view` to see a interactive GUI of the possible rostopics

        #Adi: subscribing to the zivid depth topics (Da Vinci)
        self.sub_rgb_raw = rospy.Subscriber("zivid_camera/color/image_color", Image, self.callback_rgb_raw)
        self.sub_depth_raw = rospy.Subscriber("zivid_camera/depth/image_raw", Image, self.callback_depth_raw)
        self._sub_info = rospy.Subscriber("zivid_camera/color/camera_info", CameraInfo, self.callback_cam_info)

        #Code for the Fetch:
        #self.sub_rgb_raw = rospy.Subscriber('head_camera/rgb/image_raw', Image, self.callback_rgb_raw)
        #self.sub_depth_raw = rospy.Subscriber("head_camera/depth/image_raw", Image, self.callback_depth_raw)
        #self._sub_info = rospy.Subscriber("head_camera/rgb/camera_info", CameraInfo, self.callback_cam_info)

    def callback_rgb_raw(self, data):
        try:
            self.img_rgb_raw = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.color_time_stamped = data.header.stamp
            self.is_updated = True
        except CvBridgeError as e:
            rospy.logerr(e)
            print(e)

    def callback_depth_raw(self, data):
        try:
            # you do not need to anything more than this to get the depth imgmsg as a cv2
            # make sure that you are using the right encoding
            self.img_depth_raw = self.bridge.imgmsg_to_cv2(data, '32FC1')
        except CvBridgeError as e:
            rospy.logerr(e)

    def callback_cam_info(self, data):
        try:
            self._info = data
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

    def read_color_data(self):
        return self.img_rgb_raw

    def read_depth_data(self):
        return self.img_depth_raw

    def read_info_data(self):
        return self._info

    def set_color_none(self):
        """Added to allow ENTER in `run.py` to get one single image from zivid camera."""
        self.img_rgb_raw = None

    def set_depth_none(self):
        """Added to allow ENTER in `run.py` to get one single image from zivid camera."""
        self.img_depth_raw = None


def depth_to_3ch(d_img, cutoff_min, cutoff_max):
    """Process depth images the same as in the ISRR 2019 paper.

    Only applies if we're using depth images.
    EDIT: actually we're going to add a min cutoff!
    """
    w,h = d_img.shape
    n_img = np.zeros([w, h, 3])
    d_img = d_img.flatten()

    # Instead of this:
    #d_img[d_img>cutoff] = 0.0
    # Do this? The cutoff_max means beyond the cutoff, pixels become white.
    #d_img[ d_img>cutoff_max ] = 0.0
    d_img[ d_img>cutoff_max ] = cutoff_max
    d_img[ d_img<cutoff_min ] = cutoff_min
    print('max/min depth after cutoff: {:.3f} {:.3f}'.format(np.max(d_img), np.min(d_img)))

    d_img = d_img.reshape([w,h])
    for i in range(3):
        n_img[:, :, i] = d_img
    return n_img


def depth_3ch_to_255(d_img):
    """Process depth images the same as in the ISRR 2019 paper.

    Only applies if we're using depth images.
    EDIT: actually we're going to add a min cutoff!
    """
    # Instead of this:
    #d_img = 255.0/np.max(d_img)*d_img

    # Do this:
    d_img = d_img * (255.0 / (np.max(d_img)-np.min(d_img)) )  # pixels within a 255-interval
    d_img = d_img - np.min(d_img)                             # pixels actually in [0,255]

    # Now do the inpainting?

    d_img = np.array(d_img, dtype=np.uint8)
    for i in range(3):
        d_img[:, :, i] = cv2.equalizeHist(d_img[:, :, i])
    return d_img    


def process_img_for_net(img, ix=0, iy=0):
    """Do any sort of processing of the image for the neural network.

    Only does cropping and re-sizing, for now.

    For example, we definitely need to crop, and we may want to do some
    filtering or blurring to smoothen the texture. Our network uses images of
    size (100,100) but as long as we process it and then make sure it has the
    same height and width it'll be fine -- the net class has a resize command as
    a backup.

    Processing should be done before the cropping, because doing filtering after
    cropping results in very blurry images (the filters cover a wider range).
    """
    # First component 'height', second component 'width'.  Decrease 'height'
    # values to get images higher up, decrease 'width' to make it move left.

    # IF CHANGING THESE, CHECK THAT INPAINTING IS CONSISTENT. I do this with
    # inpaint_x and inpaint_y, or ix and iy.
    img = img[135-ix:635-ix, 580-iy:1080-iy]
    assert img.shape[0] == img.shape[1]

    img = cv2.resize(img, (100, 100))
    return img


if __name__=='__main__':
    rgbd = RGBD(init_camera=True)
    i = 0
    nb_images = 1
    head = "/home/davinci0/seita/dvrk_python/tmp"

    # For real physical robot experiments, use these values in `config.py`.
    CUTOFF_MIN = 0.800
    CUTOFF_MAX = 0.905
    IN_PAINT = True

    while i < nb_images:
        print(os.listdir(head))
        num = len([x for x in os.listdir(head) if 'c_img_crop' in x])
        print('current index is at: {}'.format(num))

        d_img = None
        c_img = None
    
        print('querying the depth ...')
        while d_img is None:
            d_img = rgbd.read_depth_data()
        print('querying the RGB ...')
        while c_img is None:
            c_img = rgbd.read_color_data()
        
        # Check for NaNs.
        nb_items = np.prod(np.shape(c_img))
        nb_not_nan = np.count_nonzero(~np.isnan(c_img))
        print('RGB image shape {}, has {} items'.format(c_img.shape, nb_items))
        print('  num NOT nan: {}, or {:.2f}%'.format(nb_not_nan, nb_not_nan/float(nb_items)*100))
        nb_items = np.prod(np.shape(d_img))
        nb_not_nan = np.count_nonzero(~np.isnan(d_img))
        print('depth image shape {}, has {} items'.format(d_img.shape, nb_items))
        print('  num NOT nan: {}, or {:.2f}%'.format(nb_not_nan, nb_not_nan/float(nb_items)*100))

        # We fill in NaNs with zeros.
        c_img[np.isnan(c_img)] = 0
        d_img[np.isnan(d_img)] = 0

        # Images are 1200 x 1920, with 3 channels (well, we force for depth).
        assert d_img.shape == (1200, 1920), d_img.shape
        assert c_img.shape == (1200, 1920, 3), c_img.shape

        # BUT we can call `inpaint` which will fill in the zero pixels!
        if IN_PAINT:
            d_img = U.inpaint_depth_image(d_img)

        # Check depth image. Also, we have to tune the cutoff.
        # The depth is clearly in METERS, but I think it's hard to get an
        # accurate cutoff, sadly.
        print('\nAfter NaN filtering of the depth images ...')
        print('  max: {:.3f}'.format(np.max(d_img)))
        print('  min: {:.3f}'.format(np.min(d_img)))
        print('  mean: {:.3f}'.format(np.mean(d_img)))
        print('  medi: {:.3f}'.format(np.median(d_img)))
        print('  std: {:.3f}'.format(np.std(d_img)))

        # I think we need a version with and without the cropped for depth.
        d_img_crop = process_img_for_net(d_img)
        print('\nAfter NaN filtering of the depth images ... now for the CROPPED image:')
        print('  max: {:.3f}'.format(np.max(d_img_crop)))
        print('  min: {:.3f}'.format(np.min(d_img_crop)))
        print('  mean: {:.3f}'.format(np.mean(d_img_crop)))
        print('  medi: {:.3f}'.format(np.median(d_img_crop)))
        print('  std: {:.3f}'.format(np.std(d_img_crop)))
        print('')

        # Let's process depth. Note that we do the cropped vs noncropped
        # separately, so the cropped one shouldn't have closer noisy values from
        # the dvrk arm affecting its calculations.
        d_img      = depth_to_3ch(d_img,      cutoff_min=CUTOFF_MIN, cutoff_max=CUTOFF_MAX)
        d_img_crop = depth_to_3ch(d_img_crop, cutoff_min=CUTOFF_MIN, cutoff_max=CUTOFF_MAX)
        d_img      = depth_3ch_to_255(d_img)
        d_img_crop = depth_3ch_to_255(d_img_crop)

        c_img_crop = process_img_for_net(c_img)
        assert c_img_crop.shape[0] == c_img_crop.shape[1], c_img.shape

        # Try blurring depth, bilateral recommends 9 for offline applications
        # that need heavy blurring. The two sigmas were 75 by default.
        d_img_crop_blur = cv2.bilateralFilter(d_img_crop, 9, 100, 100)
        #d_img_crop_blur = cv2.medianBlur(d_img_crop_blur, 5)

        c_tail           = "{}_c_img.png".format(str(num).zfill(2))
        d_tail           = "{}_d_img.png".format(str(num).zfill(2))
        c_tail_crop      = "{}_c_img_crop.png".format(str(num).zfill(2))
        d_tail_crop      = "{}_d_img_crop.png".format(str(num).zfill(2))
        d_tail_crop_blur = "{}_d_img_crop_blur.png".format(str(num).zfill(2))

        c_img_path           = join(head, c_tail)
        d_img_path           = join(head, d_tail)
        c_img_path_crop      = join(head, c_tail_crop)
        d_img_path_crop      = join(head, d_tail_crop)
        d_img_path_crop_blur = join(head, d_tail_crop_blur)

        cv2.imwrite(c_img_path,           c_img)
        cv2.imwrite(d_img_path,           d_img)
        cv2.imwrite(c_img_path_crop,      c_img_crop)
        cv2.imwrite(d_img_path_crop,      d_img_crop)
        cv2.imwrite(d_img_path_crop_blur, d_img_crop_blur)

        print('  just saved: {}'.format(c_img_path))
        print('  just saved: {}'.format(d_img_path))
        print('  just saved: {}'.format(c_img_path_crop))
        print('  just saved: {}'.format(d_img_path_crop))
        print('  just saved: {}'.format(d_img_path_crop_blur))
        i += 1
    
    #rospy.spin()
