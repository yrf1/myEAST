import os
import cv2
import numpy as np
import lanms
import model
import logging
import tensorflow as tf
from shapely.geometry import Polygon
from icdar import restore_rectangle

tf.app.flags.DEFINE_string('checkpoint_path', 'east_icdar2015_resnet_v1_50_rbox/', '')
FLAGS = tf.app.flags.FLAGS

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def detect(score_map, geo_map, score_map_thresh=0.1, box_thresh=0.005, nms_thres=0.25):
    '''

    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, color=np.array((255,0,0)))
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes	

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def checkIOU(boxA, boxB):
        boxA = Polygon([(boxA[0,0], boxA[0,1]), (boxA[1,0], boxA[1,1]), (boxA[2,0], boxA[2,1]), (boxA[3,0], boxA[3,1])])
        boxB = Polygon([(int(float(boxB[0])),int(float(boxB[1])-float(boxB[3]))), (int(float(boxB[0])+float(boxB[2])), int(float(boxB[1])-float(boxB[3]))), (int(float(boxB[0])+float(boxB[2])), int(float(boxB[1]))), (int(float(boxB[0])), int(float(boxB[1])))])
        if (boxA.is_valid == False):
             return False
        intersection = boxA.intersection(boxB).area
        union = float(boxA.area + boxB.area - intersection)
        return intersection / union > 0.5

def main(argv=None):
	with tf.get_default_graph().as_default():
		input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
		f_score, f_geometry = model.model(input_images, is_training=False)
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
                saver = tf.train.Saver(variable_averages.variables_to_restore())
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		    ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
                    model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
		    saver.restore(sess, model_path)
		    logging.info(model_path)
		    with open('Data/cropped_annotations_train.txt', 'r') as f:
		        annotation_file = f.readlines()
		    count_right = 0
		    count_wrong = 0
		    count_posNotDetected = 0
		    idx = 0
		    for line in annotation_file:
		        if len(line)>1 and line[:6] == './crop':
		            logging.info(line)
			    file_name = "Data/cropped_img_train/"+line[14:].split(".tiff",1)[0]+".tiff"
		            num_true_pos = int(annotation_file[idx+1])
		            count_right_cache = 0
        		    logging.info(file_name)
		            im = cv2.imread(file_name)[:, :, ::-1]
		            im_resized, (ratio_h, ratio_w) = resize_image(im)
		            score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
		            boxes = detect(score_map=score, geo_map=geometry)
		            if boxes is not None:
		                boxes = boxes[:, :8].reshape((-1, 4, 2))
		                boxes[:, :, 0] /= ratio_w
		                boxes[:, :, 1] /= ratio_h
		                for box in boxes:
		                    box = sort_poly(box.astype(np.int32))
		                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
		                        continue
		                    count_wrong += 1
		                    for i in range(num_true_pos):
		                        if (checkIOU(box, annotation_file[idx+2+i].split(" ")) == True):
		                            count_right_cache += 1
		                            count_wrong -= 1
		            count_posNotDetected += num_true_pos - count_right_cache
		            count_right += count_right_cache
		        idx += 1
		    precision = (float) (count_right) / (float) (count_right + count_wrong)  # TP / TP + FP
		    recall = (float) (count_right) / (float) (count_right + count_posNotDetected)  # TP / TP + FN
		    fscore = 2 * (precision * recall) / (precision + recall)
		    logging.info("precision {:.4f}, recall {:.4f}, fscore {:.4f}".format(precision, recall, fscore))

if __name__ == '__main__':
    logging.basicConfig(filename='logexperiment.log', level=logging.INFO)
    tf.app.run()
