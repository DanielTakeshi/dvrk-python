"""Shared methods, to be loaded in other code.
"""
import os
import sys
import cv2
import time
import numpy as np
from os import path
from os.path import join
import datetime


# Useful constants.
ESC_KEYS = [27, 1048603]
MILLION = float(10**6)


def rad_to_deg(rad):
    return np.array(rad) * 180./np.pi


def deg_to_rad(deg):
    return np.array(deg) * np.pi/180.


def normalize(v):
    norm=np.linalg.norm(v, ord=2)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def save_image_numbers(head, img, indicator=None, debug=False):
    """Save image in a directory, but numbered at the end.
    
    Example, indicator might be `c_img`. Note: if we import os.path like `from
    os import path`, then please avoid name conflicts!
    """
    if indicator is None:
        nb = len([x for x in os.listdir(head) if '.png' in x])
        new_path = join(head, 'img_{}.png'.format(str(nb).zfill(4)))
    else:
        nb = len([x for x in os.listdir(head) if indicator in x])
        new_path = join(head, '{}_{}.png'.format(indicator, str(nb).zfill(4)))
    if debug:
        print('saving to: {}'.format(new_path))
    cv2.imwrite(new_path, img) 


def get_date():
    """Make save path for whatever agent we are training.
    """
    date = '{}'.format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )
    return date


def call_wait_key(nothing=None, force_exit=False):
    """Call this like: `utils.call_wait_key( cv2.imshow(...) )`."""
    key = cv2.waitKey(0)
    if key in ESC_KEYS:
        print("Pressed ESC key. Terminating program...")
        if force_exit:
            sys.exit()
        else:
            return True
    return False


def inpaint_depth_image(d_img, ix=0, iy=0):
    """Inpaint depth image on raw depth values.

    Only import code here to avoid making them required if we're not inpainting.

    Also, inpainting is slow, so crop some irrelevant values. But BE CAREFUL!
    Make sure any cropping here will lead to logical consistency with the
    processing in `camera.process_img_for_net` later. For now we crop the 'later
    part' of each dimension, which still leads to > 2x speed-up. The
    window size is 3 which I think means we can get away with a pixel difference
    of 3 when cropping but to be safe let's add a bit more, 50 pix to each side.

    For `ix` and `iy` see `camera.process_img_for_net`, makes inpainting faster.
    """
    d_img = d_img[ix:685,iy:1130]
    from perception import (ColorImage, DepthImage)
    print('now in-painting the depth image (shape {}), ix, iy = {}, {}...'.format(
            d_img.shape, ix, iy))
    start_t = time.time()
    d_img = DepthImage(d_img)
    d_img = d_img.inpaint()     # inpaint, then get d_img right away
    d_img = d_img.data          # get raw data back from the class
    cum_t = time.time() - start_t
    print('finished in-painting in {:.2f} seconds'.format(cum_t))
    return d_img


def calculate_coverage(c_img, bounding_dims=(11,87,10,88), rgb_cutoff=170, display=False):
    """
    Given precomputed constant preset locations that represent the corners in a
    clockwise order, it computes the percent of pixels that are above a certain
    threshold in that region which represents the percent coverage.

    The bounding dimensions represent (min_x, max_x, min_y, max_y). To decrease
    height, confusingly, decrease min_x and max_x.  The default bounding_dims
    work well empirically. COLOR IMAGES ONLY!!

    Returns a value between [0,1].
    """
    min_x, max_x, min_y, max_y = bounding_dims
    substrate = c_img[min_x:max_x,min_y:max_y,:]
    is_not_covered = np.logical_and(np.logical_and(substrate[:,:,0] > rgb_cutoff,
        substrate[:,:,1] > rgb_cutoff), substrate[:,:,2] > rgb_cutoff)

    # can display this fake image to sanity check this method
    fake_image = np.array(is_not_covered * 255, dtype = np.uint8)
    if display:
        cv2.imshow("1", substrate)
        cv2.imshow("2", fake_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 1.0 - (np.sum(is_not_covered) / float(is_not_covered.size))


def load_mapping_table(row_board, column_board, file_name, cloth_height=0.005):
    """Load the mapping table which we need to map from neural net to action.

    The mapping table looks like this:

        nx,ny,rx,ry,rz

    Where `nx,ny` are coordinates w.r.t. the background plane, of which the
    cloth lies on top. Numbers range from (-1,1) and should be discretized in
    the mapping table. The `rx,ry,rz` are the x,y,z positions w.r.t. the robot's
    frame, and were derived by moving the robot gripper to that position over a
    checkerboard. Note that rotation is not included in the mapping table.

    :param row_board: number of rows.
    :param column_board: number of columns.
    :param file_name: name of the calibration file
    :param cloth_height: height offset, we add to the z values from the data.
    :return: data from calibration
    """
    assert os.path.exists(file_name), \
            'The file does not exist: {}'.format(file_name)
    data_default = np.loadtxt(file_name, delimiter=',')

    cnt = 0
    for i in range(row_board):
        for j in range(column_board):
            data_default[cnt, 0] = -1 + j * 0.4
            data_default[cnt, 1] = -1 + i * 0.4
            data_default[cnt, 4] = data_default[cnt, 4] + cloth_height
            cnt += 1
    data = data_default

    # Daniel: a bit confused about this, but it seems necessary to convert to
    # PSM space. See `transform_CB2PSM`.
    data_square = np.zeros((row_board + 1, column_board + 1, 5))
    for i in range(row_board):
        for j in range(column_board):
            data_square[i, j, :] = data[column_board * j + i, 0:5]

    for i in range(row_board):
        data_square[i, column_board, :] = data_square[i, column_board - 1, :]
    for j in range(column_board):
        data_square[row_board, j] = data_square[row_board - 1, j]

    return data_square


def transform_CB2PSM(x, y, row_board, col_board, data_square):
    """Minho's code, for calibation, figure out the PSM coordinates.

    Parameters (x,y) should be in [-1,1] (if not we clip it) and represent
    the coordinate range over the WHITE CLOTH BACKGROUND PLANE (or a
    'checkboard' plane). We then convert to a PSM coordinate.

    Uses bilinear interpolation.

    :param row_board: number of rows.
    :param col_board: number of columns.
    :param data_square: data from calibration.
    """
    if x>1: x=1.0
    if x<-1: x=-1.0
    if y>1:  y=1.0
    if y<-1: y=-1.0

    for i in range(row_board):
        for j in range(col_board):
            if x == data_square[row_board-1, j, 0] and y == data_square[i, col_board-1, 1]: # corner point (x=1,y=1)
                return data_square[row_board-1,col_board-1,2:5]
            else:
                if x == data_square[row_board-1, j, 0]:  # border line of x-axis
                    if data_square[i, j, 1] <= y and y < data_square[i, j + 1, 1]:
                        y1 = data_square[row_board-1, j, 1]
                        y2 = data_square[row_board-1, j+1, 1]
                        Q11 = data_square[row_board-1, j, 2:5]
                        Q12 = data_square[row_board-1, j+1, 2:5]
                        return (y2-y)/(y2-y1)*Q11 + (y-y1)/(y2-y1)*Q12
                elif y == data_square[i, col_board-1, 1]:  # border line of y-axis
                    if data_square[i, j, 0] <= x and x < data_square[i + 1, j, 0]:
                        x1 = data_square[i, col_board-1, 0]
                        x2 = data_square[i+1, col_board-1, 0]
                        Q11 = data_square[i, col_board-1, 2:5]
                        Q21 = data_square[i+1, col_board-1, 2:5]
                        return (x2-x)/(x2-x1)*Q11 + (x-x1)/(x2-x1)*Q21
                else:
                    if data_square[i,j,0] <= x and x < data_square[i+1,j,0]:
                        if data_square[i,j,1] <= y and y < data_square[i,j+1,1]:
                            x1 = data_square[i, j, 0]
                            x2 = data_square[i+1, j, 0]
                            y1 = data_square[i, j, 1]
                            y2 = data_square[i, j+1, 1]
                            Q11 = data_square[i, j, 2:5]
                            Q12 = data_square[i, j+1, 2:5]
                            Q21 = data_square[i+1, j, 2:5]
                            Q22 = data_square[i+1, j+1, 2:5]
                            if x1==x2 or y1==y2:
                                return []
                            else:
                                return 1/(x2-x1)/(y2-y1)*(Q11*(x2-x)*(y2-y) + Q21*(x-x1)*(y2-y) + Q12*(x2-x)*(y-y1) + Q22*(x-x1)*(y-y1))


def move_p_from_net_output(x, y, dx, dy, row_board, col_board, data_square, p,
                           debug=False, only_do_pick=False):
    """Minho's code, for calibration, processes policy network output.

    Be careful, the x,y coordinate from the neural net refers to a coordinate
    range of [-1,1] in the x and y directions. Thus, (x,y) = (-1,-1) is the
    BOTTOM LEFT CORNER.

    However, in simulation, we first converted (x,y) into the range [0,1] by
    dividing by two (to get values in [-0.5,0.5]) and then adding 0.5. Then,
    given x and y values that ranged from [0,1], we deduced a dx and dy such
    that ... when we apply the action, dx and dy independently 'adjust' x and y.
    So it is indeed (x+dx) and (y+dy). To convert this to our case, it should be
    as simple as doubling the dx and dy values.
    
    It's a bit trick to understand by reading gym-cloth code, because I first
    convert dx and dy into other values, and then I repeatdly do motions until
    the full length is achieved.
    
    :params (x, y, dx, dy): outputs from the neural network, all in [-1,1].
    :param row_board: number of rows.
    :param col_board: number of columns.
    :param data_square: data from calibration, from `utils.load_mapping_table`.
    :param p: An instance of `dvrkClothSim`.
    """
    assert -1 <= x <= 1, x
    assert -1 <= y <= 1, y
    assert -1 <= dx <= 1, dx
    assert -1 <= dy <= 1, dy

    # Find the targets, and then get pose w.r.t. PSM.
    targ_x = x + 2*dx
    targ_y = y + 2*dy
    pickup_pos = transform_CB2PSM(x,
                                  y,
                                  row_board,
                                  col_board,
                                  data_square)
    release_pos_temp = transform_CB2PSM(targ_x,
                                        targ_y,
                                        row_board,
                                        col_board,
                                        data_square)

    release_pos = np.array([release_pos_temp[0], release_pos_temp[1]])
    if debug:
        print('pickup position wrt PSM: {}'.format(pickup_pos))
        print('release position wrt PSM: {}'.format(release_pos))
    # just checking if the ROS input is fine
    # user_input = raw_input("Are you sure the values to input to the robot arm?(y or n)")
    # if user_input == "y":

    p.move_pose_pickup(pickup_pos, release_pos, 0, 'rad', only_do_pick=only_do_pick)


def print_means(images):
    average_img = np.zeros((100,100,3))
    for img in images:
        average_img += img
    a_img = average_img / len(images)
    print('ch 1: {:.1f} +/- {:.1f}'.format(np.mean(a_img[:,:,0]), np.std(a_img[:,:,0])))
    print('ch 2: {:.1f} +/- {:.1f}'.format(np.mean(a_img[:,:,1]), np.std(a_img[:,:,1])))
    print('ch 3: {:.1f} +/- {:.1f}'.format(np.mean(a_img[:,:,2]), np.std(a_img[:,:,2])))
    return a_img


def single_means(img, depth):
    if depth:
        print('Depth img:')
    else:
        print('Color img:')
    print('  ch 1: {:.1f} +/- {:.1f}'.format(np.mean(img[:,:,0]), np.std(img[:,:,0])))
    print('  ch 2: {:.1f} +/- {:.1f}'.format(np.mean(img[:,:,1]), np.std(img[:,:,1])))
    print('  ch 3: {:.1f} +/- {:.1f}'.format(np.mean(img[:,:,2]), np.std(img[:,:,2])))


def _adjust_gamma(image, gamma=1.0):
    """For darkening images.

    Builds a lookup table mapping the pixel values [0, 255] to their
    adjusted gamma values.

    https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 \
            for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


if __name__ == "__main__":
    # This should not be called normally! Just for testing and debugging.

    # Directory of images.
    img_paths = sorted(
            [join('tmp',x) for x in os.listdir('tmp/') if 'c_img' in x and '.png' in x]
    )
    images = [cv2.imread(x) for x in img_paths]
    img_paths_d = sorted(
            [join('tmp',x) for x in os.listdir('tmp/') if 'd_img' in x and '.png' in x]
    )
    images_d = [cv2.imread(x) for x in img_paths_d]

    # Compute means, compare with simulated data.
    print('depth across all data in directory:')
    _ = print_means(images_d)
    print('color across all data in directory:')
    _ = print_means(images)
   
    # Inspect coverage.
    nb_imgs = len(img_paths)
    print('num images: {}'.format(nb_imgs))

    # Ignore if I want to skip.
    if False:
        for idx,(img,fname) in enumerate(zip(images,img_paths)):
            coverage = calculate_coverage(img, display=True)
            print('  image {} at {} has coverage {:.2f}'.format(idx, fname, coverage*100))


    # Daniel NOTE! I was using this for debugging if we wanted to forcibly
    # adjust pixel values to get them in line with the training data.
    # Fortunately it seems close enough that we don't have to change anything.

    # Save any modified images. DEPTH here. Works well.
    if False:
        for idx,(img,fname) in enumerate(zip(images_d,img_paths_d)):
            print('  on image {}'.format(fname))
            single_means(img, depth=True)

            meanval = np.mean(img)
            target_meanval = 135.0
            img = np.minimum(np.maximum( img+(target_meanval-meanval), 0), 255)
            img = np.uint8(img)
            print('after correcting:')
            single_means(img, depth=True)
            
            savepath = fname.replace('tmp','tmp2')
            print('saving to: {}\n'.format(savepath))
            cv2.imwrite(savepath, img)

    # Save any modified images. RGB here. NOTE: doesn't work quite well
    if False:
        for idx,(img,fname) in enumerate(zip(images,img_paths)):
            print('  on image {}'.format(fname))
            single_means(img, depth=False)

            mean_0 = np.mean(img[:,:,0])
            mean_1 = np.mean(img[:,:,1])
            mean_2 = np.mean(img[:,:,2])

            means = np.array([mean_0, mean_1, mean_2])
            targets = np.array([155.0, 110.0, 85.0])
            img = np.minimum(np.maximum( img+(targets-means), 0), 255)
            img = np.uint8(img)
            print('after correcting:')
            single_means(img, depth=False)
            
            savepath = fname.replace('tmp','tmp2')
            print('saving to: {}\n'.format(savepath))
            cv2.imwrite(savepath, img)

    # Now for RGB gamma corrections.
    if False:
        for idx,(img,fname) in enumerate(zip(images,img_paths)):
            print('  on image {}'.format(fname))
            single_means(img, depth=False)

            img = _adjust_gamma(img, gamma=1.5)
            print('after correcting:')
            single_means(img, depth=False)
            
            savepath = fname.replace('tmp','tmp2')
            print('saving to: {}\n'.format(savepath))
            cv2.imwrite(savepath, img)

    # Try de-noising.
    if True:
        print('\n\nTRYING DE-NOISING\n')
        # Depth images are type uint8.
        for idx,(img,fname) in enumerate(zip(images_d,img_paths_d)):
            print('  on image {}'.format(fname))
            img = cv2.fastNlMeansDenoising(img,None,7,7,21)
            savepath = fname.replace('tmp','tmp2')
            print('saving to: {}\n'.format(savepath))
            cv2.imwrite(savepath, img)
        for idx,(img,fname) in enumerate(zip(images,img_paths)):
            print('  on image {}'.format(fname))
            img = cv2.fastNlMeansDenoisingColored(img,None,7,7,7,21)
            savepath = fname.replace('tmp','tmp2')
            print('saving to: {}\n'.format(savepath))
            cv2.imwrite(savepath, img)

