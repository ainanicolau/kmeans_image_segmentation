#!/usr/bin/env python3

import argparse
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def parse_arguments():
    """
    Parses command line options.
    """
    parser = argparse.ArgumentParser(description="Implementation of k-means "
                                                 "algorithm")
    parser.add_argument("--k", type=int, default=3, help="number of clusters")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="number of samples")
    parser.add_argument("--amplitude", type=int, default=5,
                        help="amplitude of the samples' position")
    args = parser.parse_args()
    return args


def generate_centroids(k, samples):
    """
    Assigns a random index to each sample and create k centroids which position
    is the mean of the samples with the corresponding index.
    """
    centroid_index = np.random.randint(k, size=len(samples))

    centroid_list = []
    for centroid in range(k):
        centroid_samples = samples[np.where(centroid_index == centroid)]
        centroid_list.append(np.mean(centroid_samples, axis=0))
    centroids = np.array(centroid_list)

    plots = []
    plot_points(centroids, samples, centroid_index, plots)

    return centroids, plots


def create_clusters(centroids, samples, plots):
    """
    Computes the distance between each sample and centroid to assign the samples
    to the closest centroid index.
    """
    dist = []
    for centroid in centroids:
        new_dist = np.linalg.norm(centroid - samples, axis=1)
        dist.append(new_dist)

    centroid_index = np.argmin(dist, axis=0)

    plot_points(centroids, samples, centroid_index, plots)

    return centroid_index, plots


def update_centroid(centroids, samples, centroid_index, plots):
    """
    Computes the new position for the centroids with the mean of the samples
    that belong them. If none of the centroids is changing the position the
    function returns the value to stop the algorithm.
    """
    stop = True
    for centroid in range(len(centroids)):
        new_position = np.mean(samples[np.where(centroid_index == centroid)],
                               axis=0)
        if not (np.sum(new_position - centroids[centroid]) == 0):
            stop = False
        centroids[centroid] = new_position

    plot_points(centroids, samples, centroid_index, plots)

    return centroids, stop, plots


def plot_points(centroids, samples, centroid_index, plots):
    """
    Plots the centroids and samples per iteration.
    """
    plot_points = []
    colors = cm.rainbow(np.linspace(0, 1, len(centroids)))
    for index in range(len(centroids)):
        plot_points.append(plt.plot(centroids[index, 0], centroids[index, 1],
                                    color=colors[index], marker='o')[0])

    if centroid_index.any():
        for index in range(len(samples)):
            plot_points.append(plt.plot(samples[index, 0], samples[index, 1],
                                        color=colors[centroid_index[index]],
                                        marker='.')[0])
    else:
        plot_points.append(plt.plot(samples[:, 0], samples[:, 1], 'k.')[0])

    plots.append(plot_points)

    return plots


def save_animation(fig, plots):
    """
    Creates an animated gif for the different iterations of the algorithm.
    """
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Aina'), bitrate=1800)
    ani = animation.ArtistAnimation(
        fig, plots, interval=50, repeat_delay=3000, blit=True)
    ani.save('kmeans.mp4', writer=writer)


def main():
    args = parse_arguments()

    # Create an array of samples to apply kmeans to
    samples = np.random.random([args.num_samples, 2])*args.amplitude

    fig = plt.figure()
    plt.axis([-1, np.max(samples), -1, (np.max(samples))])

    # Initialization step
    centroids, plots = generate_centroids(args.k, samples)

    stop = False
    while (not stop):
        # Assignment step
        centroid_index, plots = create_clusters(centroids, samples, plots)

        # Update step
        centroids, stop, plots = update_centroid(
            centroids, samples, centroid_index, plots)

    save_animation(fig, plots)


if __name__ == '__main__':
    main()
