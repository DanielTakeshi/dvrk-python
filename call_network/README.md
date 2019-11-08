# Network Loading

We train our neural networks with the `baselines-fork` package, and then we
deploy it here. The `baselines-fork` requires Python 3.5+, whereas ROS still
needs Python 2.7, so we save the image in Python 2.7, and then load the image
with this code, which runs in a separate tab.

## Installation

First, make a Python 3.5+ virtualenv. The commands differ from Python 3.5 and
3.6 but you can find them online easily. Here's Python 3.5 for example:

```
virtualenv path/to/venv --python=python3
```

Activate the virtualenv:

```
. path/to/venv/bin/activate
```

These subsequent commands will assume your virtualenv is activated.

This is not required, but I strongly suggest installing iPython with: `pip
install ipython`.

Install Tensorflow 1.13.1 with the GPU:

```
pip install tensorflow-gpu==1.13.1
```

or if there's no GPU:

```
pip install tensorflow==1.13.1
```

Note that davinci0 currently does not have CUDA installed.

Other versions of Tensorflow may work, but for the sake of our sanity, let's be
consistent with our versions.

Install `baselines-fork`. Clone the repository, go into the package directory,
and then run the installation script there.

```
git clone https://github.com/BerkeleyAutomation/baselines-fork.git
cd baselines-fork
pip install -e .
```

To test loading, run `python image_manip/load_net.py` with `do_test=False` (in
the main method) and you should see it load the weights. You can also set
`do_test=True` in the main method of `image_manip/load_net.py`, and put images
in the correct directory to read from.



## Usage

In a separate terminal tab, in a Python3 virtualenv, run:

```
python call_network/load_net.py
```

and then run the dvrk code. The script should loop forever, periodically
checking for an image in the target directory.

