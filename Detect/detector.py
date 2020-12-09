from mrcnn.model import log
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn.config import Config
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
from astropy import units as u

# Ignore warnings
warnings.filterwarnings("ignore")


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """

    # Give the configuration a recognizable name
    NAME = "craters"
    BACKBONE = "resnet101"  # default resnet101

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels, from 4,8
    RPN_NMS_THRESHOLD = 0.7
    MEAN_PIXEL = [165.32, 165.32, 165.32]

    # POST_NMS_ROIS_TRAINING = 2000
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.

    TRAIN_ROIS_PER_IMAGE = 300
    MAX_GT_INSTANCES = 400
    DETECTION_MAX_INSTANCES = 400

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 160

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 16
    # Additional Setting by user
    DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_NMS_THRESHOLD = 0.4


def get_ax(rows=1, cols=1, size=15):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def inspect_results(img, bboxs, color="red"):
    b = bboxs
    image = img.copy()
    for i in range(b.shape[0]):

        d1, d2 = b[i, :][1] - b[i, :][3], b[i, :][0] - b[i, :][2]
        d1, d2 = abs(d1), abs(d2)

        r = (d1 + d2) // 4
        x_c, y_c = (b[i, :][1] + b[i, :][3]
                    ) // 2, (b[i, :][0] + b[i, :][2]) // 2

        center_coordinates = (x_c, y_c)
        radius = r
        if color == "red":
            color = (255, 0, 0)
        elif color == "green":
            color = (0, 255, 0)

        thickness = 2
        cv2.circle(image, center_coordinates, radius, color, thickness)

    return image


def diff_bb(gt_boundingboxes, bounding_boxes):
    global x, y, differenza
    y = gt_boundingboxes.shape[0]
    x = bounding_boxes.shape[0]

    if y != 0:
        differenza = (x - y) / y
        return differenza


def delinvalidvalues(x):
    x = x[x != 0]
    x = x[np.isfinite(x)]
    return x


def runinference(iter):
    image_ids = np.random.choice(dataset_test.image_ids, 20)

    global P_ARRAY, R_ARRAY, F1_ARRAY, mAP_ARRAY

    APs = []
    P_ARRAY = []
    R_ARRAY = []
    F1_ARRAY = []
    mAP_ARRAY = []

    for i in range(iter):

        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
                dataset_test, inference_config, image_id
            )
            molded_images = np.expand_dims(
                modellib.mold_image(image, inference_config), 0
            )
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps = utils.compute_ap(
                gt_bbox,
                gt_class_id,
                gt_mask,
                r["rois"],
                r["class_ids"],
                r["scores"],
                r["masks"],
            )
            APs.append(AP)

            # print("mAP: ", np.mean(APs))

            P = np.mean(precisions)
            R = np.mean(recalls)
            F1 = 2 * P * R / (P + R)
            mAP = np.mean(APs)

            P_ARRAY = np.append(P_ARRAY, P)
            R_ARRAY = np.append(R_ARRAY, R)
            F1_ARRAY = np.append(F1_ARRAY, F1)
            mAP_ARRAY = np.append(mAP_ARRAY, mAP)

    P_ARRAY = delinvalidvalues(P_ARRAY)
    R_ARRAY = delinvalidvalues(R_ARRAY)
    F1_ARRAY = delinvalidvalues(F1_ARRAY)
    mAP_ARRAY = delinvalidvalues(mAP_ARRAY)

    return P_ARRAY, R_ARRAY, F1_ARRAY, mAP_ARRAY


def preprocess(img):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    image = clahe.apply(image)
    # plt.imshow(image, cmap='gray')
    image.shape
    image3channel = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # plt.imshow(image3channel)
    return image3channel


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def match_prep(b, scores, img_dim):
    vstack = np.zeros(3)
    for i in range(b.shape[0]):
        d1, d2 = b[i, :][1] - b[i, :][3], b[i, :][0] - b[i, :][2]
        d1, d2 = abs(d1), abs(d2)

        r = (d1 + d2) / 4
        x_c, y_c = (b[i, :][1] + b[i, :][3]) / 2, (b[i, :][0] + b[i, :][2]) / 2

        radius = r
        brick = [x_c, y_c, radius]
        # Post-Pro:
        score = scores[i]
        if radius > img_dim/3:
            if score > 0.6+0.00114*radius:
                vstack = np.vstack((vstack, brick))
        else:
            vstack = np.vstack((vstack, brick))

    vstack = vstack[1:, :]
    return vstack


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85


inference_config = InferenceConfig()
inference_config.USE_MINI_MASK = False
MODEL_DIR = os.path.abspath("")
# Recreate the model in inference mode
model = modellib.MaskRCNN(
    mode="inference", config=inference_config, model_dir=MODEL_DIR
)

model.load_weights(
    "Pesi/30-100_norm/mask_rcnn_craters.h5",
    by_name=True,
)
# print("Weights Loaded!")

config = ShapesConfig()
# config.display()


def detect(img):
    img_prepro = preprocess(img)
    results = model.detect([img_prepro], verbose=0)
    r = results[0]
    b = r["rois"]
    scores = r["scores"]
    carters_detected = match_prep(b, scores, img.shape[0])

    return carters_detected


def img_plus_crts(img, craters_det, color="red"):
    b = craters_det
    image = img.copy()
    for i in range(b.shape[0]):

        r = b[i][2]
        x_c, y_c = b[i][0], b[i][1]

        center_coordinates = (int(x_c), int(y_c))
        radius = int(r)
        if color == "red":
            color = (255, 0, 0)
        elif color == "green":
            color = (0, 255, 0)

        thickness = 2
        cv2.circle(image, center_coordinates, radius, color, thickness)

    return image


def main():
    pass


if __name__ == "__main__":
    main()
