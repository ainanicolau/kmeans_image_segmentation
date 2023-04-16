import numpy as np
import PIL


def __generate_centroids(k, im):
    """
    Generates k centroids by assigning a random index to each pixel in the
    input image and computing the mean of all pixels with the corresponding
    index.

    Args:
        k (int): The number of centroids to generate.
        im (numpy.ndarray): The input image as a numpy array.

    Returns:
        numpy.ndarray: A numpy array of shape (k, 3) containing the RGB values
            of the k centroids.
    """
    centroid_index = np.random.randint(k, size=len(im))

    centroid_list = []
    for centroid in range(k):
        centroid_list.append(
            np.mean(im[np.where(centroid_index == centroid)], axis=0)
        )
    centroids = np.array(centroid_list)

    return centroids


def __create_clusters(centroids, im):
    """
    Assigns each pixel in the input image to the closest centroid index based
    on the Euclidean distance between the pixel and each centroid.

    Args:
        centroids (numpy.ndarray): A numpy array of shape (k, 3) containing the
            RGB values of the k centroids.
        im (numpy.ndarray): The input image as a numpy array.

    Returns:
        numpy.ndarray: A numpy array of shape (n,) containing the centroid
            index for each pixel in the input image.
    """
    dist = []
    for centroid in centroids:
        new_dist = np.linalg.norm(centroid - im, axis=1)
        dist.append(new_dist)

    centroid_index = np.argmin(dist, axis=0)

    return centroid_index


def __update_centroid(centroids, im, centroid_index):
    """
    Computes the new position for each centroid by taking the mean of the RGB
    values of the pixels that belong to it. If none of the centroids change
    position during the update step, the function returns a boolean value to
    signal that the algorithm should stop.

    Args:
        centroids (numpy.ndarray): A numpy array of shape (k, 3) containing the
            RGB values of the k centroids.
        im (numpy.ndarray): The input image as a numpy array.
        centroid_index (numpy.ndarray): A numpy array containing the index of
            the closest centroid for each pixel in the input image.

    Returns:
        Tuple: A tuple containing the updated centroid positions as a numpy
            array of shape (k, 3) and a boolean value indicating whether any of
            the centroids changed position during the update step.
    """
    stop = True
    for centroid in range(len(centroids)):
        if im[np.where(centroid_index == centroid)].any():
            new_position = np.mean(
                im[np.where(centroid_index == centroid)], axis=0
            )
        else:
            new_position = centroids[centroid]
        if not (np.sum(new_position - centroids[centroid]) == 0):
            stop = False
        centroids[centroid] = new_position

    return centroids, stop


def __create_segmented_image(centroids, centroid_index, im_shape):
    """
    Creates a new image with pixel values based on the centroids and the
    centroid index.

    Args:
        centroids (numpy.ndarray): An array containing the RGB values of the
            centroids.
        centroid_index (numpy.ndarray): An array containing the index of the
            centroid for each pixel in the original image.
        im_shape (tuple): A tuple containing the shape of the original image.

    Returns:
        A PIL.Image object containing the new segmented image.
    """
    segmented_array = centroids[centroid_index]
    segmented_image = np.array(
        segmented_array.reshape(im_shape[0], im_shape[1], 3)
    )
    im = PIL.Image.fromarray(np.uint8(segmented_image))

    return im


def run(im, k):
    """
    Runs the k-means algorithm on the input image to create a segmented image
    with a specified number of clusters.

    Args:
        im (numpy.ndarray): The input image to segment, represented as a NumPy
            array.
        k (int): The number of clusters to create using k-means.

    Returns:
        numpy.ndarray: The segmented image, represented as a NumPy array.
    """
    im_original = np.array(im)
    im = im_original.reshape(im_original.shape[0] * im_original.shape[1], 3)

    # Initialization step
    centroids = __generate_centroids(k, im)

    stop = False
    while not stop:
        # Assignment step
        centroid_index = __create_clusters(centroids, im)

        # Update step
        centroids, stop = __update_centroid(centroids, im, centroid_index)

    im = __create_segmented_image(centroids, centroid_index, im_original.shape)

    return im
