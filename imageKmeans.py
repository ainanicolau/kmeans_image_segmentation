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
        # maxValues = np.max(im, axis=0)
        # index = np.random.randint(len(im), size=k)
        # centroids = im[index]

        maxValues = np.max(im, axis=0)
        centroidsX = np.random.random(k)*maxValues[0]
        centroidsY = np.random.random(k)*maxValues[1]
        centroidsZ = np.random.random(k)*maxValues[2]
        centroids = np.array([centroidsX,centroidsY,centroidsZ]).transpose()

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
        plt.figure('Segmented Image')
        segmentedArray = centroids[centroidIndex]
        segmentedImage = np.array(
            segmentedArray.reshape(imShape[0],imShape[1],3))
        plt.imshow(segmentedImage/256)


def main():       
        args = parseArguments()

        imOriginal = np.array(PIL.Image.open(args.im))
        im = imOriginal.reshape(imOriginal.shape[0]*imOriginal.shape[1],3)

        plt.figure('Initial Data')
        ax = plt.axes(projection='3d')

        # ax.scatter3D(im[::30,0],im[::30,1],im[::30,2],color='k',marker='.')
        for pixel in range(0,len(im),30):
             ax.plot3D([im[pixel,0]],[im[pixel,1]],[im[pixel,2]],
                 color = (im[pixel,0]/256.,im[pixel,1]/256.,im[pixel,2]/256.),
                 marker = '.')

        # Initialization step
        centroids = generateCentroids(args.k, im)
        ax.scatter3D(centroids[:,0],centroids[:,1],centroids[:,2], 
            color='k',marker='o')

        i=0
        stop = False
        while (i<5):#(not stop):
                # Assignment step
                centroidIndex = createClusters(centroids, im)

                # Update step
                centroids, stop = updateCentroid(centroids, im, centroidIndex)
                i=i+1

        segmentedImage = createSegmentedImage(
            centroids, centroidIndex, imOriginal.shape)


        plt.figure('Final Result')
        ax = plt.axes(projection='3d')

        colors = cm.rainbow(np.linspace(0, 1, len(centroids)))
        for index in range(len(centroids)):
            ax.scatter3D(
                [centroids[index,0]],
                [centroids[index,1]],
                [centroids[index,2]], 
                color=colors[index], 
                marker='o')

        for index in range(len(centroids)):
            clusterPixels = im[np.where(centroidIndex==index)]
            ax.scatter3D(
                clusterPixels[::30,0],
                clusterPixels[::30,1],
                clusterPixels[::30,2], 
                color=colors[index], 
                marker='.')


        plt.figure('Final Result Real Colours')
        ax = plt.axes(projection='3d')

        colors = cm.rainbow(np.linspace(0, 1, len(centroids)))
        for index in range(len(centroids)):
            ax.plot3D(
                [centroids[index,0]],
                [centroids[index,1]],
                [centroids[index,2]],
                color='k',
                marker='o')

        for index in range(len(centroids)):
            clusterPixels = im[np.where(centroidIndex==index)]
            ax.scatter3D(
                clusterPixels[::30,0],
                clusterPixels[::30,1],
                clusterPixels[::30,2],
                color=(
                    centroids[index,0]/256.,
                    centroids[index,1]/256.,
                    centroids[index,2]/256.),
                marker='.')

        plt.show()


if __name__ == '__main__':
        main()
