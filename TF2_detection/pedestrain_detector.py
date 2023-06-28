import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np


class PedestrianDetector():
  def __init__(self,path_to_model,path_to_label_map,threshold=0.4):
    self.path_to_model=path_to_model
    self.path_to_label_map=path_to_label_map
    self.threshold=threshold
    self.category_index=label_map_util.create_category_index_from_labelmap(path_to_label_map,
                                                                    use_display_name=True)
    tf.keras.backend.clear_session()
    self.detect_fn=tf.saved_model.load(path_to_model)
  def detect_from_img(self,image_path):
    image = cv2.imread(image_path)
    input_tensor=tf.convert_to_tensor(image)
    input_tensor=input_tensor[tf.newaxis, ...]
    detections = self.detect_fn(input_tensor)
    num_detections=int(detections.pop('num_detections'))
    image_with_detections = image.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_with_detections,
          detections['detection_boxes'][0,:num_detections].numpy(),
          detections['detection_classes'][0,:num_detections].numpy().astype(np.int64),
          detections['detection_scores'][0,:num_detections].numpy(),
          category_index=self.category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=100,
          min_score_thresh=self.threshold,
          agnostic_mode=False)
    return image_with_detections
    # cv2.imshow(image_with_detections)
# detector=PedestrianDetector(path_to_model=r'TF2_detection\exported-models\my-faster-rcnn\saved_model',
#                                         path_to_label_map=r'TF2_detection\exported-models\my-faster-rcnn\label_map.pbtxt')
# detector.detect_from_img(r'C:\Users\91805\TF2_detection\uploads\CrossWalk_(5465840138).jpg')
