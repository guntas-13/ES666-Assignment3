import pdb
import glob
import cv2
import os
import numpy as np
from src.GuntasSaran.utils import compute_blending_weights, normalize_weights, transform_corners, get_shift_matrix_left_size, get_shift_matrix_right_size, cylindrical_warp
from src.GuntasSaran.homographies import estimate_homography

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        images = [cv2.imread(im) for im in all_images]
        
        if images[0].shape == (490, 653, 3) or images[0].shape == (487, 730, 3):
            images = [cylindrical_warp(im, 800) for im in images]
        
        if len(images) % 2 == 0:
            reference_idx = len(images) // 2 - 1
        else:
            reference_idx = len(images) // 2
        
        homography_matrix_list = self.get_homographies(images, reference_idx)
        
        blended_image_left = images[0].astype(np.float64)
        total_weight_left = None
        shift_matrix_left = np.eye(3)
        for i in range(1, reference_idx + 1):
            blended_image_left, total_weight_left, shift_matrix_left = self.blend_images_left(
                blended_image_left, images[i], homography_matrix_list[i - 1], total_weight_left, shift_matrix_left)
        
        if images[0].shape == (1329, 2000, 3):
            blended_image_right = images[-1].astype(np.float64)
        else:
            blended_image_right = images[-1]
        total_weight_right = None
        shift_matrix_right = np.eye(3)
        
        for i in range(len(images) - 2, reference_idx, -1):
            blended_image_right, total_weight_right, shift_matrix_right = self.blend_images_right(
                blended_image_right, images[i], homography_matrix_list[i], total_weight_right, shift_matrix_right)
        
    
        H = homography_matrix_list[reference_idx]
        
        corners, min_x, min_y, max_x, max_y = transform_corners(blended_image_right, H @ np.linalg.inv(shift_matrix_right))
        corners1, min_x1, min_y1, max_x1, max_y1 = transform_corners(blended_image_left, np.linalg.inv(shift_matrix_left))
        
        final_canvas_width = int(max_x - min_x1)
        final_canvas_height = int(max(max_y, max_y1) - min(min_y, min_y1))
        
        final_shift_matrix_left = np.array([[1, 0, -min_x1], [0, 1, -min(min_y1, min_y)], [0, 0, 1]])
        final_shift_matrix_right = np.array([[1, 0, (final_canvas_width - max_x)], [0, 1, -min(min_y1, min_y)], [0, 0, 1]])
        
        final_shift_matrix_left = final_shift_matrix_left.astype(np.float64)
        final_shift_matrix_right = final_shift_matrix_right.astype(np.float64)
        
        blended_image_left_final = cv2.warpPerspective(blended_image_left, final_shift_matrix_left @ np.linalg.inv(shift_matrix_left), (final_canvas_width, final_canvas_height))
        blended_image_right_final = cv2.warpPerspective(blended_image_right, final_shift_matrix_right @ H @ np.linalg.inv(shift_matrix_right), (final_canvas_width, final_canvas_height))
        
        total_weight_left_final = cv2.warpPerspective(total_weight_left, final_shift_matrix_left @ np.linalg.inv(shift_matrix_left), (final_canvas_width, final_canvas_height))
        
        if total_weight_right is None:
            total_weight_right_final = compute_blending_weights(blended_image_right, final_shift_matrix_right @ H, (final_canvas_width, final_canvas_height))
        else:
            total_weight_right_final = cv2.warpPerspective(total_weight_right, final_shift_matrix_right @ H @ np.linalg.inv(shift_matrix_right), (final_canvas_width, final_canvas_height))
        
        total_weight_left_final_normalised, total_weight_right_final_normalised, _ = normalize_weights(total_weight_left_final, total_weight_right_final)
        
        blended_image_left_final = blended_image_left_final.astype(np.float64)
        blended_image_right_final = blended_image_right_final.astype(np.float64)
        total_weight_left_final_normalised = total_weight_left_final_normalised.astype(np.float64)
        total_weight_right_final_normalised = total_weight_right_final_normalised.astype(np.float64)

        stitched_image = (blended_image_left_final * total_weight_left_final_normalised + blended_image_right_final * total_weight_right_final_normalised)
        stitched_image = np.clip(stitched_image / stitched_image.max() * 255, 0, 255).astype(np.uint8)
        
        return stitched_image, homography_matrix_list 
    
    
    def get_homographies(self, images, reference_idx, useOpenCV = False):
        """Takes in a list of images and returns a list of homographies between the images.

        Args:
            images (List): List of images to compute homographies for.
            reference_idx (int): Index of the reference image.
            useOpenCV (bool, optional): Defaults to False. Whether to use OpenCV's homography estimation or my own.

        Returns:
            List: List of homographies between the images.
        """
        homographies = []
        for i in range(1, reference_idx + 1):
            H = estimate_homography(images[i - 1], images[i], useOpenCV)
            homographies.append(H)
        
        for i in range(reference_idx, len(images) - 1):
            H = estimate_homography(images[i + 1], images[i], useOpenCV)
            homographies.append(H)
        
        return homographies
    
    def blend_images_left(self, image1, image2, H, total_weight, shift_matrix):
        """Blend two images together from the left.

        Args:
            image1 (np.ndarray): Blended Image from Left.
            image2 (np.ndarray): Image Reference.
            H (np.ndarray): Homography matrix between the images.
            total_weight (np.ndarray): Total weight of the images.
            shift_matrix (np.ndarray): Shift matrix.

        Returns:
            np.ndarray: Blended image.
        """
        # initially pass total_weight as None and shift_matrix as np.eye(3)
        
        shift_matrix_new, size = get_shift_matrix_left_size(image1, image2, H @ np.linalg.inv(shift_matrix))
        shift_matrix_new = shift_matrix_new.astype(np.float64)
        
        image1_warped = cv2.warpPerspective(image1, shift_matrix_new @ H @ np.linalg.inv(shift_matrix), size)
        image2_warped = cv2.warpPerspective(image2, shift_matrix_new, size)
        
        if total_weight is None:
            total_weight = compute_blending_weights(image1, shift_matrix_new @ H, size)
        else:
            total_weight = cv2.warpPerspective(total_weight, shift_matrix_new @ H @ np.linalg.inv(shift_matrix), size)
        
        weight1 = total_weight.copy()
        weight2 = compute_blending_weights(image2, shift_matrix_new, size)
        
        weight1_normalised, weight2_normalised, total_weight = normalize_weights(weight1, weight2)
        
        image1_warped = image1_warped.astype(np.float64)
        image2_warped = image2_warped.astype(np.float64)
        weight1_normalised = weight1_normalised.astype(np.float64)
        weight2_normalised = weight2_normalised.astype(np.float64)
        
        blended_image = (image1_warped * weight1_normalised + image2_warped * weight2_normalised)
        blended_image = np.clip(blended_image / blended_image.max() * 255, 0, 255).astype(np.uint8)
        
        return blended_image, total_weight, shift_matrix_new
    
    
    def blend_images_right(self, image_blended, image_ref, H, total_weight, shift_matrix):
    
        # image1 is the reference image and image2 is the image to be blended
        # initially pass total_weight as None and shift_matrix as np.eye(3)
        
        shift_matrix_new, size = get_shift_matrix_right_size(image_blended, image_ref, H @ np.linalg.inv(shift_matrix))
        shift_matrix_new = shift_matrix_new.astype(np.float64)
        warped_image_ref = cv2.warpPerspective(image_ref, shift_matrix_new, size)
        warped_image_blended = cv2.warpPerspective(image_blended, shift_matrix_new @ H @ np.linalg.inv(shift_matrix), size)
        
        if total_weight is None:
            total_weight = compute_blending_weights(image_blended, shift_matrix_new @ H, size)
        else:
            total_weight = cv2.warpPerspective(total_weight, shift_matrix_new @ H @ np.linalg.inv(shift_matrix), size)
        
        weight_blended = total_weight.copy()
        weight_ref = compute_blending_weights(image_ref, shift_matrix_new, size)
        
        weight_blended_normalised, weight_ref_normalised, total_weight = normalize_weights(weight_blended, weight_ref)
        
        warped_image_blended = warped_image_blended.astype(np.float64)
        warped_image_ref = warped_image_ref.astype(np.float64)
        weight_blended_normalised = weight_blended_normalised.astype(np.float64)
        weight_ref_normalised = weight_ref_normalised.astype(np.float64)
        
        blended_image = (warped_image_blended * weight_blended_normalised + warped_image_ref * weight_ref_normalised)
        # print(np.unique(blended_image))
        blended_image = np.clip(blended_image / blended_image.max() * 255, 0, 255).astype(np.uint8)
        
        return blended_image, total_weight, shift_matrix_new