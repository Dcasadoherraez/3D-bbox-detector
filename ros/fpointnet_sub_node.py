#!/usr/bin/env python
import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensor_msgs.msg import Image, PointCloud2, PointField
from vision_msgs.msg import BoundingBox2DArray
from std_msgs.msg import String
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2
import message_filters
import kitti.kitti_util as utils
from models.frustum_pointnets import FrustumPointNetv1

from std_msgs.msg import Header
import cv2
import rospy
import operator
import numpy as np
import torch

DATASET_ROOT = "/hdd/Thesis/SiameseTracker/dataset/KITTI/training"
IMG_TAG = "image_02"
SEQ = "0000"
PC_TAG = "velodyne"
IMG_SEQUENCE = os.path.join(IMG_TAG, SEQ)
PC_SEQUENCE = os.path.join(PC_TAG, SEQ)
SEQ_CALIB_PATH = os.path.join(DATASET_ROOT, "calib", "0000.txt")
IMAGE_DATASET_PATH = os.path.join(DATASET_ROOT, IMG_SEQUENCE)
PC_DATASET_PATH = os.path.join(DATASET_ROOT, PC_SEQUENCE)
LABELS_PATH = os.path.join(DATASET_ROOT, "label_02", SEQ + ".txt")
PUB_RATE = 10

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs

class KittiSubscriber:
    def __init__(self) -> None:
        self.load_pretrained = ''
        self.n_classes = 3
        self.bridge = CvBridge()
        rospy.init_node('fpointnet_subscriber', anonymous=True)

        ts = message_filters.TimeSynchronizer([
            message_filters.Subscriber('/image_publisher', Image),
            message_filters.Subscriber('/depth_publisher', PointCloud2),
            message_filters.Subscriber('/gt_bb_publisher_3D', BoundingBoxArray),
            message_filters.Subscriber('/gt_bb_publisher_2D', BoundingBox2DArray),
        ], queue_size=1)
        
        ts.registerCallback(self.detect_objects)

        rospy.spin()
    
    def detect_objects(self, image, depth, gt_3d, gt_2d):
        print('Hi')

    def test_from_rgb_detection(self, image, depth, gt_3d_list, gt_2d_list):
        ''' Test frustum pointnets with GT 2D boxes.
        Write test results to KITTI format label files.
        todo (rqi): support variable number of points.
        '''

        '''
        batch_data:[32, 2048, 4], pts in frustum
        batch_label:[32, 2048], pts ins seg label in frustum
        batch_center:[32, 3],
        batch_hclass:[32],
        batch_hres:[32],
        batch_sclass:[32],
        batch_sres:[32,3],
        batch_rot_angle:[32],
        batch_one_hot_vec:[32,3],
        '''

        FrustumPointNet = FrustumPointNetv1(n_classes=self.n_classes).cuda()

        # load pre-trained model
        if self.load_pretrained:
            ckpt = torch.load(self.load_pretrained)
            FrustumPointNet.load_state_dict(ckpt['model_state_dict'])

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = data

        data_dicts = {}

        bs = batch_data.shape[0]
        
        data_dicts['point_cloud'] = batch_data.transpose(2,1).float().cuda()
        data_dicts['rot_angle'] = batch_rot_angle.float().cuda()
        data_dicts['box3d_center'] = batch_center.float().cuda()
        data_dicts['size_class'] = batch_sclass.long().cuda()
        data_dicts['size_residual'] = batch_sres.float().cuda()
        data_dicts['angle_class']  = batch_hclass.long().cuda()
        data_dicts['angle_residual'] = batch_hres.float().cuda()
        data_dicts['one_hot'] = batch_one_hot_vec.float().cuda()
        data_dicts['seg'] = batch_label.float().cuda()

        FrustumPointNet = model.eval()

        logits, seg_label, box3d_center, box3d_center_label, stage1_center, \
        heading_scores, heading_residual_normalized, heading_residual, \
        heading_class_label, heading_residual_label, size_scores, \
        size_residual_normalized, size_residual, size_class_label, \
        size_residual_label = \
            FrustumPointNet(data_dicts)



        # Pre-trained Model
        # pth = torch.load(model_path)
        # FrustumPointNet.load_state_dict(pth['model_state_dict'])

        # 1. Load data
        batch_data, \
        batch_rot_angle, \
        batch_rgb_prob, \
        batch_one_hot_vec = data

        # return point_set, rot_angle, self.prob_list[index], one_hot_vec
        batch_data = batch_data.transpose(2, 1).float().cuda()
        # batch_label = batch_label.float().cuda()
        # batch_center = batch_center.float().cuda()
        # batch_hclass = batch_hclass.float().cuda()
        # batch_hres = batch_hres.float().cuda()
        # batch_sclass = batch_sclass.float().cuda()
        # batch_sres = batch_sres.float().cuda()
        batch_rot_angle = batch_rot_angle.float().cuda()
        batch_rgb_prob = batch_rgb_prob.float().cuda()
        batch_one_hot_vec = batch_one_hot_vec.float().cuda()

        # 2. Eval one batch
        model = model.eval()
        logits, mask, stage1_center, center_boxnet, \
        heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals, center = \
            model(batch_data, batch_one_hot_vec)
        # logits:[32, 1024, 2] , mask:[32, 1024]

        logits = logits.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        center_boxnet = center_boxnet.cpu().detach().numpy()
        # stage1_center = stage1_center.cpu().detach().numpy()#
        center = center.cpu().detach().numpy()
        heading_scores = heading_scores.cpu().detach().numpy()
        # heading_residuals_normalized = heading_residuals_normalized.cpu().detach().numpy()
        heading_residuals = heading_residuals.cpu().detach().numpy()
        size_scores = size_scores.cpu().detach().numpy()
        size_residuals = size_residuals.cpu().detach().numpy()
        # size_residuals_normalized = size_residuals_normalized.cpu().detach().numpy()#
        batch_rot_angle = batch_rot_angle.cpu().detach().numpy()

        # 5. Compute and write all Results
        batch_output = np.argmax(logits, 2)  # mask#torch.Size([32, 1024])
        batch_center_pred = center  # _boxnet#torch.Size([32, 3])
        batch_hclass_pred = np.argmax(heading_scores, 1)  # (32,)
        batch_hres_pred = np.array([heading_residuals[j, batch_hclass_pred[j]] \
                                    for j in range(batch_data.shape[0])])  # (32,)
        # batch_size_cls,batch_size_res
        batch_sclass_pred = np.argmax(size_scores, 1)  # (32,)
        batch_sres_pred = np.vstack([size_residuals[j, batch_sclass_pred[j], :] \
                                        for j in range(batch_data.shape[0])])  # (32,3)

        # batch_scores
        batch_seg_prob = softmax(logits)[:, :, 1]  # (32, 1024, 2) ->(32, 1024)
        batch_seg_mask = np.argmax(logits, 2)  # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1)  # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask, 1)  # B,
        heading_prob = np.max(softmax(heading_scores), 1)  # B
        size_prob = np.max(softmax(size_scores), 1)  # B,
        # batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)

        mask_max_prob = np.max(batch_seg_prob * batch_seg_mask, 1)
        batch_scores = mask_max_prob




if __name__ == '__main__':
    print("Starting frustrum pointnet subscriber node...")

    pub = KittiSubscriber()
 
