#!/usr/bin/env python3

import argparse
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
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


def generate_centroids(k, im, ax):
    """
    Assigns a random index to each pixel and create k centroids which position
    is the mean of the pixel with the corresponding index. Plots the centroids
    into the Initial Data figure.
    """
    centroid_index = np.random.randint(k, size=len(im))

    centroid_list = []
    for centroid in range(k):
        centroid_pixels = im[np.where(centroid_index == centroid)]
        centroid_list.append(np.mean(centroid_pixels, axis=0))
    centroids = np.array(centroid_list)

    ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='k',
                 marker='o')

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
            new_position = np.mean(im[np.where(centroid_index == centroid)],
                                   axis=0)
        else:
            new_position = centroids[centroid]
        if not (np.sum(new_position - centroids[centroid]) == 0):
                stop = False
        centroids[centroid] = new_position

    return centroids, stop


def create_segmented_image(centroids, centroid_index, im_shape):
    """
    Plots the resulting segmented image.
    """
    plt.figure('Segmented Image')
    segmented_array = centroids[centroid_index]
    segmented_image = np.array(segmented_array.reshape(im_shape[0],
                                                       im_shape[1], 3))
    plt.imshow(segmented_image / 256)


def plot_initial_data(im):
    """
    Plots the original image pixels into a 3D RGB space.
    """
    plt.figure('Initial Data')
    ax = plt.axes(projection='3d')

    for pixel in range(0, len(im), 30):
        color = (im[pixel, 0] / 256., im[pixel, 1] / 256., im[pixel, 2] / 256.)
        ax.plot3D([im[pixel, 0]], [im[pixel, 1]], [im[pixel, 2]],
                  color=color, marker='.')

    return ax


def plot_results(title, im, centroids, centroid_index, ax,
                 color_centroids=None, color_pixels=None):
    """
    Plots the resulting clusters
    """
    plt.figure(title)
    ax = plt.axes(projection='3d')

    color_map = cm.rainbow(np.linspace(0, 1, len(centroids)))

    for index in range(len(centroids)):
        color = color_centroids or color_map[index]
        ax.scatter3D([centroids[index, 0]], [centroids[index, 1]],
                     [centroids[index, 2]], color=color, marker='o')

    for index in range(len(centroids)):
        if color_pixels == 'real':
            color = (centroids[index, 0] / 256., centroids[index, 1] / 256.,
                     centroids[index, 2] / 256.)
        else:
            color = color_map[index]
        cluster_pixels = im[np.where(centroid_index == index)]
        ax.scatter3D(cluster_pixels[::30, 0], cluster_pixels[::30, 1],
                     cluster_pixels[::30, 2], color=color, marker='.')


def main():
    args = parse_arguments()

    im_original = np.array(PIL.Image.open(args.im))
    im_lenght = im_original.shape[0] * im_original.shape[1]
    im = im_original[:, :, 0:3].reshape(im_lenght, 3)

    ax = plot_initial_data(im)

    # Initialization step
    centroids = generate_centroids(args.k, im, ax)

    stop = False
    while (not stop):
        # Assignment step
        centroid_index = create_clusters(centroids, im)

        # Update step
        centroids, stop = update_centroid(centroids, im, centroid_index)

    segmented_image = create_segmented_image(centroids, centroid_index,
                                             im_original.shape)

    plot_results('Final Result', im, centroids, centroid_index, ax)
    plot_results('Final Result Real Colours', im, centroids, centroid_index,
                 ax, 'k', 'real')

    plt.show()


if __name__ == '__main__':
        main()
