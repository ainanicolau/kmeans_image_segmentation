#!/usr/bin/env python3

import argparse
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import PIL
from mpl_toolkits import mplot3d


def parseArguments():
        parser = argparse.ArgumentParser(
                description="Implementation of k-means algorithm")
        parser.add_argument(
                "--k", type=int, default= 3,
                help="number of clusters")
        parser.add_argument(
                "--im", type=str,
                help="image to segmentate")
        args = parser.parse_args()
        return args


def generateCentroids(k, im):
        centroidIndex = np.random.randint(k, size=len(im))

        centroidList = []
        for centroid in range(k):
            centroidList.append(np.mean(im[np.where(centroidIndex==centroid)],axis=0))
        centroids = np.array(centroidList)

        return centroids


def createClusters(centroids, im):
        dist=[]
        for centroid in centroids:
                newDist = np.linalg.norm(centroid-im, axis=1)
                dist.append(newDist)

        centroidIndex = np.argmin(dist, axis=0)

        return centroidIndex


def updateCentroid(centroids, im, centroidIndex):
        stop = True
        for centroid in range(len(centroids)):
                if (im[np.where(centroidIndex==centroid)].any()):
                    newPosition = np.mean(
                        im[np.where(centroidIndex==centroid)],axis=0)
                else:
                    newPosition = centroids[centroid]
                if not (np.sum(newPosition - centroids[centroid])==0):
                        stop = False
                centroids[centroid] = newPosition

        return centroids, stop


def createSegmentedImage(centroids, centroidIndex, imShape):
        segmentedArray = centroids[centroidIndex]
        segmentedImage = np.array(
            segmentedArray.reshape(imShape[0],imShape[1],3))
        im = PIL.Image.fromarray(np.uint8(segmentedImage))
        return(im)


def run(im, k):
        imOriginal = np.array(im)
        im = imOriginal.reshape(imOriginal.shape[0]*imOriginal.shape[1],3)

        # Initialization step
        centroids = generateCentroids(k, im)

        i=0
        stop = False
        while (not stop):
                # Assignment step
                centroidIndex = createClusters(centroids, im)

                # Update step
                centroids, stop = updateCentroid(centroids, im, centroidIndex)
                i=i+1

        im = createSegmentedImage(centroids, centroidIndex, imOriginal.shape)
        return(im)


def main():       
        args = parseArguments()
        run(args.im, args.k)


if __name__ == '__main__':
        main()
