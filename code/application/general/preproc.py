import numpy as np
import cv2
import os 

def preproc(img_path, input_size, swap=(2, 0, 1)):

    img_info = {"id": 0}
    if isinstance(img_path, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
    else:
            img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    img_info["ratio"] = ratio
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    padded_img = np.expand_dims(padded_img, axis=0)

    img_info = {"id": 0}
    if isinstance(img_path, str):
            img_info["file_name"] = os.path.basename(img)
    else:
            img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    img_info["ratio"] = ratio

    return img, padded_img, img_info, r