import numpy as np
import cv2

def detect_and_match_features(img1, img2):
    sift = cv2.SIFT_create()
    
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    
    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    return points1, points2, keypoints1, keypoints2, good_matches



def compute_homography(pts1, pts2):
    A = []
    for i in range(len(pts1)):
        x, y = pts1[i][0], pts1[i][1]
        x_prime, y_prime = pts2[i][0], pts2[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]



def compute_homography_ransac(points1, points2, iterations=1000, threshold=5.0):
    max_inliers = []
    points1_h = np.hstack([points1, np.ones((points1.shape[0], 1))])
    points2_h = np.hstack([points2, np.ones((points2.shape[0], 1))])

    for _ in range(iterations):
        idxs = np.random.choice(len(points1), 4, replace=False)
        pts1_sample = points1[idxs]
        pts2_sample = points2[idxs]

        H = compute_homography(pts1_sample, pts2_sample)

        projected_points = (H @ points1_h.T).T
        projected_points /= projected_points[:, 2:3]

        distances = np.linalg.norm(points2_h[:, :2] - projected_points[:, :2], axis=1)
        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            
    inlier_pts1 = points1[max_inliers]
    inlier_pts2 = points2[max_inliers]
    best_H = compute_homography(inlier_pts1, inlier_pts2)

    return best_H, len(max_inliers)


def estimate_homography(image1, image2, useOpenCV = False):
    points1, points2, _, _, _ = detect_and_match_features(image1, image2)
    if useOpenCV:
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    else:
        H, _ = compute_homography_ransac(points1, points2)
    return H