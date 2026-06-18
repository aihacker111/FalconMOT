"""Global Motion Compensation (GMC).

Estimates a 2x3 affine transform between consecutive frames so the tracker can
compensate for camera motion. Supported backends: 'none', 'sparseOptFlow'
(default), 'orb', 'sift', 'ecc'.
"""

import copy

import cv2
import numpy as np


class GMC:
    def __init__(self, method='sparseOptFlow', downscale=2, verbose=None):
        self.method = method
        self.downscale = max(1, int(downscale))

        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif self.method == 'ecc':
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1,
                                       blockSize=3, useHarrisDetector=False, k=0.04)
        elif self.method in ('none', 'None'):
            self.method = 'none'
        else:
            raise ValueError('Unknown CMC method: ' + str(method))

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame, detections=None):
        if self.method in ('orb', 'sift'):
            return self.applyFeatures(raw_frame, detections)
        if self.method == 'ecc':
            return self.applyEcc(raw_frame, detections)
        if self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, detections)
        return np.eye(2, 3)

    def applyEcc(self, raw_frame, detections=None):
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.initializedFirstFrame = True
            return H

        try:
            _, H = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except Exception:
            print('Warning: ECC transform failed. Using identity.')
        return H

    def applyFeatures(self, raw_frame, detections=None):
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        mask = np.zeros_like(frame)
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        keypoints = self.detector.detect(frame, mask)
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            self.initializedFirstFrame = True
            return H

        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)
        if len(knnMatches) == 0:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        matches, spatialDistances = [], []
        maxSpatialDistance = 0.25 * np.array([width, height])
        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prev_pt = self.prevKeyPoints[m.queryIdx].pt
                curr_pt = keypoints[m.trainIdx].pt
                sd = (prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1])
                if abs(sd[0]) < maxSpatialDistance[0] and abs(sd[1]) < maxSpatialDistance[1]:
                    spatialDistances.append(sd)
                    matches.append(m)

        mean_sd = np.mean(spatialDistances, 0)
        std_sd = np.std(spatialDistances, 0)
        inliers = (spatialDistances - mean_sd) < 2.5 * std_sd

        prevPoints, currPoints = [], []
        for i in range(len(matches)):
            if inliers[i, 0] and inliers[i, 1]:
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        if np.size(prevPoints, 0) > 4:
            H, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)
        return H

    def applySparseOptFlow(self, raw_frame, detections=None):
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H, None

        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prevFrame, frame, self.prevKeyPoints, None)

        prevPoints, currPoints = [], []
        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        if np.size(prevPoints, 0) > 4:
            H, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        return H, None
