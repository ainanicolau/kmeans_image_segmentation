#!/usr/bin/env python3

import argparse
import os
from PIL import Image

import kmeans as km


def parse_arguments():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Implementation of k-means algorithm"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="number of clusters for k-means algorithm",
    )
    parser.add_argument(
        "--im", type=str, help="path to the input image to be segmented"
    )
    args = parser.parse_args()

    return args


def main():
    """
    Parses the command line arguments and runs the k-means algorithm on the
    input image.
    """
    args = parse_arguments()

    # Open the image using the Image module
    image = Image.open(args.im)

    # Run kmean on input image
    segmented_image = km.run(image, args.k)

    # Save segmented image
    file_name, file_extension = os.path.splitext(args.im)
    output_file_path = f"{file_name}_output{file_extension}"
    segmented_image.save(output_file_path)


if __name__ == "__main__":
    main()
