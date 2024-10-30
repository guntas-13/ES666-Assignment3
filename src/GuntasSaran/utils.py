import numpy as np
from PIL import Image
import cv2


def transform(src_pts, H):
    """Transform points with a homography matrix."""
    src = np.pad(src_pts, [(0, 0), (0, 1)], constant_values=1)
    pts = np.dot(H, src.T).T
    pts = (pts / pts[:, 2].reshape(-1, 1))[:, :2]
    return pts

def bilinear_interpolate(img, x, y):
    """Perform bilinear interpolation for non-integer coordinates."""
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia * wa[:, None] + Ib * wb[:, None] + Ic * wc[:, None] + Id * wd[:, None])

def my_warp_perspective(img, H, size):
    """Warp an image with a given homography matrix and target size."""
    width, height = size

    idx_pts = np.mgrid[0:width, 0:height].reshape(2, -1).T
    
    map_pts = transform(idx_pts, np.linalg.inv(H))
    x_src, y_src = map_pts[:, 0], map_pts[:, 1]
    
    mask = (x_src >= 0) & (x_src < img.shape[1] - 1) & (y_src >= 0) & (y_src < img.shape[0] - 1)
    
    valid_x_src = x_src[mask]
    valid_y_src = y_src[mask]
    valid_idx_pts = idx_pts[mask]

    warped_pixels = bilinear_interpolate(img, valid_x_src, valid_y_src)
    
    warped = np.zeros((height, width, img.shape[2]), dtype=img.dtype)
    warped[valid_idx_pts[:, 1], valid_idx_pts[:, 0]] = warped_pixels

    return warped


def cylindrical_warp(image, focal_length):
    h, w = image.shape[:2]
    x_c, y_c = w // 2, h // 2

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    theta = (u - x_c) / focal_length
    h_cyl = (v - y_c) / focal_length

    x_hat = np.sin(theta)
    y_hat = h_cyl
    z_hat = np.cos(theta)

    x_img = (focal_length * x_hat / z_hat + x_c).astype(np.int32)
    y_img = (focal_length * y_hat / z_hat + y_c).astype(np.int32)

    valid_mask = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)

    cylindrical_img = np.zeros_like(image)
    cylindrical_img[v[valid_mask], u[valid_mask]] = image[y_img[valid_mask], x_img[valid_mask]]

    cylindrical_img = Image.fromarray(cylindrical_img)
    cylindrical_img = cylindrical_img.crop((u[valid_mask].min(), v[valid_mask].min(), u[valid_mask].max(), v[valid_mask].max()))
    cylindrical_img = np.array(cylindrical_img)

    return cylindrical_img

def single_weights_array(size: int) -> np.ndarray:
    if size % 2 == 1:
        return np.concatenate([np.linspace(0, 1, (size + 1) // 2), np.linspace(1, 0, (size + 1) // 2)[1:]])
    else:
        return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])


def single_weights_matrix(shape: tuple[int]) -> np.ndarray:
    return (single_weights_array(shape[0])[:, np.newaxis] @ single_weights_array(shape[1])[:, np.newaxis].T)


def compute_blending_weights(image, shift_matrix, final_canvas_size):
    weight = single_weights_matrix(image.shape[:2])
    warped_weight = cv2.warpPerspective(weight, shift_matrix, final_canvas_size)
    return np.repeat(warped_weight[:, :, np.newaxis], 3, axis=2)

def normalize_weights(weight1, weight2):
    total_weight = (weight1 + weight2) / (weight1 + weight2).max()
    weight1_normalized = np.divide(weight1, total_weight, where=total_weight != 0)
    weight2_normalized = np.divide(weight2, total_weight, where=total_weight != 0)
    return weight1_normalized, weight2_normalized, total_weight


def transform_corners(image, H):
    h, w = image.shape[:2]

    corners = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], dtype=np.float32)
    corners = np.expand_dims(corners, axis=1)
    
    dest_corners = cv2.perspectiveTransform(corners, H)
    dest_corners = dest_corners.reshape(4, 2)

    min_x, min_y = np.min(dest_corners, axis=0)
    max_x, max_y = np.max(dest_corners, axis=0)

    return dest_corners, min_x, min_y, max_x, max_y

def get_shift_matrix_left_size(image, image_ref, H):
    corners, min_x, min_y, max_x, max_y = transform_corners(image, H)
    shift_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    size = image_ref.shape[1] - int(min_x), image_ref.shape[0] - int(min_y)
    
    return shift_matrix, size

def get_shift_matrix_right_size(image, image_ref, H):
    corners, min_x, min_y, max_x, max_y = transform_corners(image, H)
    shift_x, shift_y = -min(min_x, 0), -min(min_y, 0)
    shift_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]])
    
    size = int(max(max_x, image_ref.shape[1]) + shift_x), int(max(max_y, image_ref.shape[0]) + shift_y)
    
    return shift_matrix, size 