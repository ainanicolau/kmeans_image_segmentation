#!/usr/bin/env python3

import argparse
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def parseArguments():
        parser = argparse.ArgumentParser(
                description="Implementation of k-means algorithm")
        parser.add_argument(
                "--k", type=int, default= 3,
                help="number of clusters")
        parser.add_argument(
                "--numSamples", type=int, default= 20,
                help="number of samples")
        parser.add_argument(
                "--amplitude", type=int, default= 5,
                help="amplitude of the samples' position")
        args = parser.parse_args()
        return args


def generateCentroids(k, samples):
        maxValues = np.max(samples, axis=0)
        centroidsX = np.random.random(k)*maxValues[0]
        centroidsY = np.random.random(k)*maxValues[1]
        centroids = np.array([centroidsX,centroidsY]).transpose()

        plots = plotPoints(centroids, samples)

        return centroids, plots


def createClusters(centroids, samples, plots):
        dist=[]
        for centroid in centroids:
                newDist = np.linalg.norm(centroid-samples, axis=1)
                dist.append(newDist)

        centroidIndex = np.argmin(dist, axis=0)

        plots = plotPoints(centroids, samples, centroidIndex, plots)

        return centroidIndex, plots


def updateCentroid(centroids, samples, centroidIndex, plots):
        stop = True
        for centroid in range(len(centroids)):
                newPosition = np.mean(
                    samples[np.where(centroidIndex==centroid)],axis=0)
                if not (np.sum(newPosition - centroids[centroid])==0):
                        stop = False
                centroids[centroid] = newPosition

        plots = plotPoints(centroids, samples, centroidIndex)

        return centroids, stop, plots


def plotPoints(centroids, samples, centroidIndex=np.array([]), plots=[]):
        plotPoints = []
        colors = cm.rainbow(np.linspace(0, 1, len(centroids)))
        for index in range(len(centroids)):
                plotPoints.append(plt.plot(
                    centroids[index,0],
                    centroids[index,1], 
                    color=colors[index], 
                    marker='o')[0])
        
        if centroidIndex.any():
                for index in range(len(samples)):
                        plotPoints.append(plt.plot(
                            samples[index,0],
                            samples[index,1], 
                            color=colors[centroidIndex[index]], 
                            marker='.')[0])
        else:
                plotPoints.append(plt.plot(samples[:,0],samples[:,1],'k.')[0])

        plots.append(plotPoints)

        return plots


def saveAnimation(fig, plots):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani = animation.ArtistAnimation(
            fig, plots, interval=50, repeat_delay=3000, blit=True)
        ani.save('kmeans.mp4', writer=writer)


def main():       
        args = parseArguments()

        samples = np.random.random([args.numSamples,2])*args.amplitude

        fig = plt.figure()
        plt.axis([-1, np.max(samples), -1, (np.max(samples))])

        # Initialization step
        centroids, plots = generateCentroids(args.k, samples)
        
        stop = False
        while (not stop):
                # Assignment step
                centroidIndex, plots = createClusters(centroids, samples, plots)

                # Update step
                centroids, stop, plots = updateCentroid(
                    centroids, samples, centroidIndex, plots)

        saveAnimation(fig, plots)


if __name__ == '__main__':
        main()
