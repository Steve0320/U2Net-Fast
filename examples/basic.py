# This example illustrates the basic use of the main Remover library. It removes
# the backgrounds from the images in the "inputs" folder, and writes them to the
# "outputs" folder.
import os

# Note that for this example, all input images must be the same dimensions. If
# your images vary in size, you will either need to pad or separate them into
# different batches.

# Before running this script ensure that you have downloaded the U2Net model to
# the examples/models directory as noted in the readme.

import os
from u2net_fast.remover import Remover


def main():

    # Set the working dir to the script file's location. By default, the remover library
    # uses the current directory to find inputs and write outputs. You could also manually
    # pass in explicit locations, but that's not necessary for a basic example.
    script_location = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_location)

    # Core call - run remover script with default values.
    x = Remover()
    x.batch_remove_background()


if __name__ == '__main__':
    main()
