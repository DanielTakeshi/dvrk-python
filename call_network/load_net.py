"""Load a trained network. Requires Python3. Put the files in the main method.

Also includes functionality to annotate the images, since I think that's needed
to show intuition in a paper.

It's simlar to the version in baselines-fork but I thought it'd be better to
use a file contained within dvrk_python, or at least as reasonably as we can.

The removal of the Python2.7 package is necessary due to ROS, see:
    https://stackoverflow.com/questions/43019951/
"""
import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import time
import pickle
import functools
import tensorflow as tf
import numpy as np
from os.path import join
import baselines.common.tf_util as U
from baselines.imit.models import Actor
import load_config as cfg
import datetime


class NetLoader:

    def __init__(self, net_file):
        """Trying to mirror the IMIT agent's loading method.

        When creating the actor, the TF is not actually created, for that we
        need placeholders and then to 'call' the actor.
        """
        self.actor = Actor(nb_actions=4, name='actor', network='cloth_cnn', use_keras=False)
        self.net_file = net_file

        # Exactly same as in the imit/imit_learner code, create actor network.
        self.observation_shape = (100, 100, 3)
        shape = (None,) + self.observation_shape
        self.obs0 = tf.placeholder(tf.int32, shape=shape, name='obs0_imgs')
        self.obs0_f_imgs = tf.cast(self.obs0, tf.float32) / 255.0
        self.actor_tf = self.actor(self.obs0_f_imgs)

        # Handle miscellaneous TF stuff.
        self.sess = U.get_session()
        self.sess.run(tf.global_variables_initializer())
        _vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print('\nThe TF variables after init:\n')
        for vv in _vars:
            print('  {}'.format(vv))
        print('\nBaselines debugging:')
        U.display_var_info(_vars)
        print('\nNow let\'s call U.load_variables() ...')

        # Our imit code loads after initializing variables.
        U.load_variables(load_path=net_file, sess=self.sess)
        self.sess.graph.finalize()

    def forward_pass(self, img, reduce=True):
        """Run a forward pass. No other processing needed, I think.
        """
        feed = {self.obs0: [img]}
        result = self.sess.run(self.actor_tf, feed_dict=feed)
        if reduce:
            result = np.squeeze(result)
        return result

    def process(self, img, debug=False):
        """Process output from raw dvrk, real world image `img`.
        """
        h ,w = self.observation_shape[0], self.observation_shape[1]
        img = cv2.resize(img, (h,w))
        return img

    def act_to_coords(self, img, act, annotate=False, img_file=None):
        """Convert action to coordinates of the white background plane

        Optionally, annotate to file.

        Returns the pick point (start) and target (ending), but unfortunately
        w.r.t. full image, and NOT the actual background plane, darn. But at
        least the forward pass itself seems to be working.  Pretty sure this
        means scaling by 50 to get our [-1,1] to [-50,50] and then adding 50
        ...
        """
        assert img.shape == self.observation_shape, img.shape
        assert img.shape[0] == img.shape[1], img.shape  # for now
        coord_min = 0
        coord_max = img.shape[0]

        # Convert from (-1,1) to the image pixels.  Note: this WILL sometimes
        # include points outside the range because we don't restrict that ---
        # but the agent should easily learn not to do that via IL or RL.
        pix_pick = (act[0] * 50 + 50,
                    act[1] * 50 + 50)
        pix_targ = ((act[0]+act[2]) * 50 + 50,
                    (act[1]+act[3]) * 50 + 50)

        # For image annotation we probably can just restrict to intervals. Also
        # convert to integers for drawing.
        pix_pick = ( int(max(min(pix_pick[0],coord_max),coord_min)),
                     int(max(min(pix_pick[1],coord_max),coord_min)) )
        pix_targ = ( int(max(min(pix_targ[0],coord_max),coord_min)),
                     int(max(min(pix_targ[1],coord_max),coord_min)) )

        if not annotate:
            return (pix_pick, pix_targ)

        # Now we annotate, save the image, and return pixels after all this.
        # AH ....we actually need this on the background plane. So it's harder
        # to interpret. Ack, we'll need a way to fit to that plane somehow ...
        # might be best to approximate it by hand?
        assert img_file is not None
        cv2.circle(img, center=pix_pick, radius=5, color=cfg.BLUE, thickness=1)
        cv2.circle(img, center=pix_targ, radius=3, color=cfg.RED, thickness=1)
        fname = img_file
        cv2.imwrite(filename=fname, img=img)

        return (pix_pick, pix_targ)


def run_test(net_l):
    """Only to test for correctness, maybe some debugging.
    """
    image_files = cfg.TEST_IMAGE_FILES

    # Where to save images (with annotated labels).
    save_img_path = join('data','tmp')
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path, exist_ok=True)

    # Iterate through images. We might as well save w/annotated images.
    for img_file in image_files:
        img = cv2.imread(img_file)
        assert img.shape == net_l.observation_shape, img.shape
        result = net_l.forward_pass(img)

        # Save the images, we may want to hav
        base = os.path.basename(os.path.normpath(img_file)).replace('resized_','')
        new_img_file = join(save_img_path, base)
        pixels = net_l.act_to_coords(img=img,
                                     act=result,
                                     annotate=True,
                                     img_file=new_img_file)
        ps, pe = pixels
        print('{}    \t {} ---> {},     from {}'.format(result, ps, pe, img_file))


if __name__ == '__main__':
    net_file = cfg.NET_FILE
    net_l = NetLoader(net_file)

    # Test if it is working, should set to False for actual dvrk experiments.
    DO_TEST = False
    if DO_TEST:
        run_test(net_l)
        sys.exit()

    # For now assume a clean directory.
    dvrk_img_paths = sorted(
        [join(cfg.DVRK_IMG_PATH,x) for x in os.listdir(cfg.DVRK_IMG_PATH) \
            if x[-4:]=='.png']
    )
    if len(dvrk_img_paths) > 0:
        print('There are {} images in {}. Please remove it/them.'.format(
                len(dvrk_img_paths), cfg.DVRK_IMG_PATH))
        print('It should be empty to start an episode.')
        sys.exit()

    # Let this code run in an infinite loop.
    print('\nNetwork loaded successfully!')
    print('Now we\'re waiting for images in: {}'.format(cfg.DVRK_IMG_PATH))
    t_start = time.time()
    beeps = 0
    nb_prev = 0
    nb_curr = 0

    while True:
        time.sleep(1)
        beeps += 1
        t_elapsed = (time.time() - t_start)
        if beeps % 5 == 0:
            print('  time: {:.2f}s (i.e., {:.2f}m)'.format(t_elapsed, t_elapsed/60.))

        # -------------------------------------------------------------------- #
        # HUGE ASSUMPTION: assume we store image sequentially and do not
        # override them. That means the images should be appearing in
        # alphabetical order in chronological order. We can compute statistics
        # about these and the actions in separate code.
        # -------------------------------------------------------------------- #
        dvrk_img_paths = sorted(
            [join(cfg.DVRK_IMG_PATH,x) for x in os.listdir(cfg.DVRK_IMG_PATH) \
                    if x[-4:]=='.png']
        )
        nb_curr = len(dvrk_img_paths)

        # Usually this equality should be happening.
        if nb_prev == nb_curr:
            continue
        if nb_prev+1 < nb_curr:
            print('Error, prev {} and curr {}, should only differ by one.'.format(
                    nb_prev, nb_curr))
        nb_prev = nb_curr

        # TODO: image processing?
        dvrk_img = cv2.imread(dvrk_img_paths[-1])
        img = net_l.process(dvrk_img)

        # Forward pass, save to directory, wait for next image.
        policy_action = net_l.forward_pass(img)
        net_results = sorted(
            [join(cfg.DVRK_IMG_PATH,x) for x in os.listdir(cfg.DVRK_IMG_PATH) \
                    if 'result_' in x and '.txt' in x]
        )
        nb_calls = len(net_results)+1
        date = '{}'.format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )
        pol = (cfg.WHICH_POLICY).split('/')[0]
        tail = 'result_{}_{}_num_{}.txt'.format(date, pol, str(nb_calls).zfill(3))
        save_path = join(cfg.DVRK_IMG_PATH, tail)
        np.savetxt(save_path, policy_action, fmt='%f')
        print('Just did action #{}, with result: {}'.format(nb_calls, policy_action))
        print('Saving to: {}'.format(save_path))
