# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 17:43:58 2020

@author: curya
"""

import numpy as np
import time

class vibe_gray:
    def __init__(self, ):
        self.width = 0
        self.height = 0
        self.numberOfSamples = 20
        self.matchingThreshold = 20
        self.matchingNumber = 2
        self.updateFactor = 16
        
        # Storage for the history
        self.historyImage = None
        self.historyBuffer = None
        self.lastHistoryImageSwapped = 0
        self.numberOfHistoryImages = 2
        
        # Buffers with random values
        self.jump = None
        self.neighbor = None
        self.position = None
        
    def AllocInit(self, image):
        print('AllocInit!')
        height, width = image.shape[:2]
        # set the parametors
        self.width = width
        self.height = height
        print(self.height, self.width)
        
        # create the historyImage
        self.historyImage = np.zeros((self.height, self.width, self.numberOfHistoryImages), np.uint8)
        for i in range(self.numberOfHistoryImages):
            self.historyImage[:, :, i] = image
            
        # create and fill the historyBuffer
        self.historyBuffer = np.zeros((self.height, self.width, self.numberOfSamples-self.numberOfHistoryImages), np.uint8)
        for i in range(self.numberOfSamples-self.numberOfHistoryImages):
            image_plus_noise = image + np.random.randint(-10, 10, (self.height, self.width))
            image_plus_noise[image_plus_noise > 255] = 255
            image_plus_noise[image_plus_noise < 0] = 0
            self.historyBuffer[:, :, i] = image_plus_noise.astype(np.uint8)
        
        # fill the buffers with random values
        size = 2 * self.width + 1 if (self.width > self.height) else 2 * self.height + 1
        self.jump = np.zeros((size), np.uint32)
        self.neighbor = np.zeros((size), np.int)
        self.position = np.zeros((size), np.uint32)
        for i in range(size):
            self.jump[i] = np.random.randint(1, 2*self.updateFactor+1)
            self.neighbor[i] = np.random.randint(-1, 3-1) + np.random.randint(-1, 3-1) * self.width
            self.position[i] = np.random.randint(0, self.numberOfSamples)
        
    
    def Segmentation(self, image):
        # segmentation_map init
        segmentation_map = np.zeros((self.height, self.width)) + (self.matchingNumber - 1)
        
        # first history image
        mask = np.abs(image - self.historyImage[:, :, 0]) > self.matchingThreshold
        segmentation_map[mask] = self.matchingNumber
        
        # next historyImages
        for i in range(1, self.numberOfHistoryImages):
            mask = np.abs(image - self.historyImage[:, :, i]) <= self.matchingThreshold
            segmentation_map[mask] = segmentation_map[mask] - 1
        
        # for swapping
        self.lastHistoryImageSwapped = (self.lastHistoryImageSwapped + 1) % self.numberOfHistoryImages
        swappingImageBuffer = self.historyImage[:, :, self.lastHistoryImageSwapped]
        
        # now, we move in the buffer and leave the historyImage
        numberOfTests = self.numberOfSamples - self.numberOfHistoryImages
        mask = segmentation_map > 0
        for i in range(numberOfTests):
            mask_ = np.abs(image - self.historyBuffer[:, :, i]) <= self.matchingThreshold
            mask_ = mask * mask_
            segmentation_map[mask_] = segmentation_map[mask_] - 1
            
            # Swapping: Putting found value in history image buffer
            temp = swappingImageBuffer[mask_].copy()
            swappingImageBuffer[mask_] = self.historyBuffer[:, :, i][mask_].copy()
            self.historyBuffer[:, :, i][mask_] = temp
        
        # simulate the exit inner loop
        mask_ = segmentation_map <= 0
        mask_ = mask * mask_
        segmentation_map[mask_] = 0
        
        # Produces the output. Note that this step is application-dependent
        mask = segmentation_map > 0
        segmentation_map[mask] = 255
        return segmentation_map.astype(np.uint8)
    
    def Update(self, image, updating_mask):
        numberOfTests = self.numberOfSamples - self.numberOfHistoryImages
        
        inner_time = 0
        t0 = time.time()
        # updating
        for y in range(1, self.height-1):
            t1 = time.time()
            shift = np.random.randint(0, self.width)
            indX = self.jump[shift]
            t2 = time.time()
            inner_time += (t2 - t1)
            #""" too slow
            while indX < self.width - 1:
                t3 = time.time()
                index = indX + y * self.width
                t4 = time.time()
                inner_time += (t4 - t3)
                if updating_mask[y, indX] == 255:
                    t5 = time.time()
                    value = image[y, indX]
                    index_neighbor = index + self.neighbor[shift]
                    y_, indX_ = int(index_neighbor / self.width), int(index_neighbor % self.width)
                    
                    if self.position[shift] < self.numberOfHistoryImages:
                        self.historyImage[y, indX, self.position[shift]] = value
                        self.historyImage[y_, indX_, self.position[shift]] = value
                    else:
                        pos = self.position[shift] - self.numberOfHistoryImages
                        self.historyBuffer[y, indX, pos] = value
                        self.historyBuffer[y_, indX_, pos] = value
                    t6 = time.time()
                    inner_time += (t6 - t5)
                t7 = time.time()
                shift = shift + 1
                indX = indX + self.jump[shift]
                t8 = time.time()
                inner_time += (t8 - t7)
            #"""
        t9 = time.time()
        # print('update: %.4f, inner time: %.4f' % (t9 - t0, inner_time))
        
        # First row
        y = 0
        shift = np.random.randint(0, self.width)
        indX = self.jump[shift]
        
        while indX <= self.width - 1:
            index = indX + y * self.width
            if updating_mask[y, indX] == 0:
                if self.position[shift] < self.numberOfHistoryImages:
                    self.historyImage[y, indX, self.position[shift]] = image[y, indX]
                else:
                    pos = self.position[shift] - self.numberOfHistoryImages
                    self.historyBuffer[y, indX, pos] = image[y, indX]
            
            shift = shift + 1
            indX = indX + self.jump[shift]
            
        # Last row
        y = self.height - 1
        shift = np.random.randint(0, self.width)
        indX = self.jump[shift]
        
        while indX <= self.width - 1:
            index = indX + y * self.width
            if updating_mask[y, indX] == 0:
                if self.position[shift] < self.numberOfHistoryImages:
                    self.historyImage[y, indX, self.position[shift]] = image[y, indX]
                else:
                    pos = self.position[shift] - self.numberOfHistoryImages
                    self.historyBuffer[y, indX, pos] = image[y, indX]
            
            shift = shift + 1
            indX = indX + self.jump[shift]
        
        # First column
        x = 0
        shift = np.random.randint(0, self.height)
        indY = self.jump[shift]
        
        while indY <= self.height - 1:
            index = x + indY * self.width
            if updating_mask[indY, x] == 0:
                if self.position[shift] < self.numberOfHistoryImages:
                    self.historyImage[indY, x, self.position[shift]] = image[indY, x]
                else:
                    pos = self.position[shift] - self.numberOfHistoryImages
                    self.historyBuffer[indY, x, pos] = image[indY, x]
            
            shift = shift + 1
            indY = indY + self.jump[shift]
            
        # Last column
        x = self.width - 1
        shift = np.random.randint(0, self.height)
        indY = self.jump[shift]
        
        while indY <= self.height - 1:
            index = x + indY * self.width
            if updating_mask[indY, x] == 0:
                if self.position[shift] < self.numberOfHistoryImages:
                    self.historyImage[indY, x, self.position[shift]] = image[indY, x]
                else:
                    pos = self.position[shift] - self.numberOfHistoryImages
                    self.historyBuffer[indY, x, pos] = image[indY, x]
            
            shift = shift + 1
            indY = indY + self.jump[shift]
            
        # The first pixel
        if np.random.randint(0, self.updateFactor) == 0:
            if updating_mask[0, 0] == 0:
                position = np.random.randint(0, self.numberOfSamples)
                
                if position < self.numberOfHistoryImages:
                    self.historyImage[0, 0, position] = image[0, 0]
                else:
                    pos = position - self.numberOfHistoryImages
                    self.historyBuffer[0, 0, pos] = image[0, 0]
        
class vibe_rgb:
    def __init__(self, ):
        pass
    
    def init(self, ):
        pass
    
    def segmentation(self, ):
        pass
    
    def update(self, ):
        pass