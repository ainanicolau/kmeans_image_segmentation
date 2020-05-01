#!/usr/bin/env python3

import argparse
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import PIL
from mpl_toolkits import mplot3d


def parse_arguments():
    """
    Parses command line options.
    """
    parser = argparse.ArgumentParser(description="Implementation of k-means "
                                                 "algorithm")
    parser.add_argument("--k", type=int, default=3, help="number of clusters")
    parser.add_argument("--im", type=str, help="image to segmentate")
    args = parser.parse_args()
    return args


def generate_centroids(k, im):
    """
    Assigns a random index to each pixel and create k centroids which position
    is the mean of the pixel with the corresponding index.
    """
    centroid_index = np.random.randint(k, size=len(im))

    centroid_list = []
    for centroid in range(k):
        centroid_list.append(np.mean(im[np.where(centroid_index == centroid)],
                                     axis=0))
    centroids = np.array(centroid_list)

    return centroids


def create_clusters(centroids, im):
    """
    Computes the distance between each sample and centroid to assign the pixels
    to the closest centroid index.
    """
    dist = []
    for centroid in centroids:
        new_dist = np.linalg.norm(centroid - im, axis=1)
        dist.append(new_dist)

    centroid_index = np.argmin(dist, axis=0)

    return centroid_index


def update_centroid(centroids, im, centroid_index):
    """
    Computes the new position for the centroids with the mean of the pixels
    that belong them. If none of the centroids is changing the position the
    function returns the value to stop the algorithm.
    """
    stop = True
    for centroid in range(len(centroids)):
        if (im[np.where(centroid_index == centroid)].any()):
            new_position = np.mean(
                im[np.where(centroid_index == centroid)], axis=0)
        else:
            new_position = centroids[centroid]
        if not (np.sum(new_position - centroids[centroid]) == 0):
                stop = False
        centroids[centroid] = new_position

    return centroids, stop


def create_segmented_image(centroids, centroid_index, im_shape):
    """
    Creates the resulting segmented image.
    """
    segmented_array = centroids[centroid_index]
    segmented_image = np.array(segmented_array.reshape(im_shape[0],
                                                       im_shape[1], 3))
    im = PIL.Image.fromarray(np.uint8(segmented_image))
    return(im)


def run(im, k):
    """
    Runs the k-means algorithm
    """
    im_original = np.array(im)
    im = im_original.reshape(im_original.shape[0]*im_original.shape[1], 3)

    # Initialization step
    centroids = generate_centroids(k, im)

    stop = False
    while (not stop):
        # Assignment step
        centroid_index = create_clusters(centroids, im)

        # Update step
        centroids, stop = update_centroid(centroids, im, centroid_index)

    im = create_segmented_image(centroids, centroid_index, im_original.shape)
    return(im)


def main():
    args = parse_arguments()
    run(args.im, args.k)


if __name__ == '__main__':
        main()
