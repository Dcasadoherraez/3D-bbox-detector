#!/usr/bin/env python
import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import operator
import rospy
import ros_numpy
import cv2
from std_msgs.msg import Header
from models.model_util import g_type2class, g_class2type, g_type2onehotclass
from models.frustum_pointnets import FrustumPointNetv1
import kitti.kitti_util as utils
import message_filters
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2, PointField
from bb_pub_node.msg import BoundingBox2D, BoundingBox2DArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from calendar import c

from wandb import Object3D

from train.provider import class2angle, from_prediction_to_label_format
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


class KittiSubscriber:
    def __init__(self) -> None:
        self.load_pretrained = "/hdd/catkin_ws/src/bb_pub_node/log/run3_carpedcyc_kitti_2022-05-05-22/acc0.641093-epoch143.pth"
        self.n_classes = 3
        self.bridge = CvBridge()
        self.calib = utils.Calibration(SEQ_CALIB_PATH)

        print('Initializing model...')
        self.model = FrustumPointNetv1(n_classes=self.n_classes)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # load pre-trained model
        if self.load_pretrained:
            print('Loading model from ', self.load_pretrained)
            ckpt = torch.load(self.load_pretrained)
            self.model.load_state_dict(ckpt['model_state_dict'])
       
        self.model.eval()

        # ROS
        rospy.init_node('fpointnet_subscriber', anonymous=True)

        # Publisher of the prediction result
        self.prediction_3d = rospy.Publisher(
            'prediction_3d_publisher', BoundingBoxArray, queue_size=1)

        # Subscriber to the data stream
        ts = message_filters.TimeSynchronizer([
            message_filters.Subscriber('/image_publisher', Image),
            message_filters.Subscriber('/depth_publisher', PointCloud2),
            message_filters.Subscriber('/gt_bb_publisher_3D', BoundingBoxArray),
            message_filters.Subscriber('/gt_bb_publisher_2D', BoundingBox2DArray),
        ], queue_size=1)

        ts.registerCallback(self.test_from_rgb_detection)

        rospy.spin()

    def pub_pred_bbox(self, predictions, timestamp):
        '''Publish predicted object bounding boxes'''
        bbox_array_3D = BoundingBoxArray()
        bbox_array_3D.boxes = []
        bbox_array_3D.header = Header()
        bbox_array_3D.header.stamp = timestamp
        bbox_array_3D.header.frame_id = "base_link"

        i = 0
        while i < len(predictions):
            tx, ty, tz, h, w, l, ry = predictions[i]

            # 3D Bounding box
            bbox_3D = BoundingBox()
            bbox_3D.header.frame_id = "base_link"
            bbox_3D.dimensions.x = w
            bbox_3D.dimensions.y = l
            bbox_3D.dimensions.z = h
            bbox_3D.pose.position.x = tx
            bbox_3D.pose.position.y = ty
            bbox_3D.pose.position.z = tz

            roll, pitch, yaw = 0, 0, ry
            qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
                np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
                np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
            qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
                np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
            qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
                np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

            bbox_3D.pose.orientation.x = qx
            bbox_3D.pose.orientation.y = qy
            bbox_3D.pose.orientation.z = qz
            bbox_3D.pose.orientation.w = qw

            bbox_array_3D.boxes.append(bbox_3D)

            i += 1

        self.prediction_3d.publish(bbox_array_3D)

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
        timestamp = image.header.stamp
        cv_image = self.bridge.imgmsg_to_cv2(
            image, desired_encoding='passthrough')
        cv_image = np.asarray(cv_image)


        bbox_2D_list = gt_2d_list.boxes
        predictions_list = []

        bbox_3D_list = gt_3d_list.boxes

        for bbox_2D, bbox_3D in zip(bbox_2D_list, bbox_3D_list):
            # Get the 3D points inside the 2D bounding box
            xmin = int(bbox_2D.center.x - bbox_2D.size_x // 2)
            xmax = int(bbox_2D.center.x + bbox_2D.size_x // 2)
            ymin = int(bbox_2D.center.y - bbox_2D.size_y // 2)
            ymax = int(bbox_2D.center.y + bbox_2D.size_y // 2)

            # cv2.rectangle(cv_image,pt1=(xmin, ymin),pt2=(xmax, ymax),color=(0, 255, 0),thickness=3,lineType=cv2.LINE_4)
            # cv2.imshow("test", cv_image)
            # cv2.waitKey(0)

            pc = ros_numpy.numpify(depth)
            pc_3d_coord = np.zeros((pc.shape[0], 4), dtype=np.float32)
            pc_3d_coord[:, 0] = pc['x']
            pc_3d_coord[:, 1] = pc['y']
            pc_3d_coord[:, 2] = pc['z']
            pc_3d_coord[:, 3] = pc['intensity']

            pc_image_coord = self.calib.project_velo_to_image(
                pc_3d_coord[:, :3])

            box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                           (pc_image_coord[:, 0] >= xmin) & \
                           (pc_image_coord[:, 1] < ymax) & \
                           (pc_image_coord[:, 1] >= ymin)

            pc_in_box_fov = pc_3d_coord[box_fov_inds, :]

            # one hot encoding
            cls_type = bbox_2D.type

            if cls_type not in ['Car', 'Pedestrian', 'Cyclist']:
                continue

            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

            input_data = {}
            pc_in_box_fov = torch.tensor(
                pc_in_box_fov, dtype=torch.float32).unsqueeze(0).transpose(2, 1).cuda()
            one_hot_vec = torch.tensor(one_hot_vec, dtype=torch.float32).cuda()

            input_data['point_cloud'] = pc_in_box_fov
            input_data['one_hot'] = one_hot_vec

            net_out = self.model(input_data)

            # net_out['logits']
            # net_out['box3d_center']
            # net_out['stage1_center']
            # net_out['heading_scores']
            # net_out['heading_residual_normalized']
            # net_out['heading_residual']
            # net_out['size_scores']
            # net_out['size_residual_normalized']
            # net_out['size_residual']

            # Get frustrum angle
            box2d_center = np.array([bbox_2D.center.x, bbox_2D.center.y])
            uvdepth = np.zeros((1, 3))
            uvdepth[0, 0:2] = box2d_center
            uvdepth[0, 2] = 20  # some random depth
            box2d_center_rect = self.calib.project_image_to_rect(uvdepth)
            frustum_angle =  -1 * np.arctan2(box2d_center_rect[0, 2], box2d_center_rect[0, 0])
            frustum_angle += np.pi / 2.0
            
            center = net_out['box3d_center'].cpu().detach().numpy()[0]
            angle_class = np.argmax(net_out['heading_scores'].cpu().detach().numpy())
            angle_res = net_out['heading_residual'].cpu().detach().numpy()[0][angle_class]

            size_class = np.argmax(net_out['size_scores'].cpu().detach().numpy())
            size_res = net_out['size_residual'].cpu().detach().numpy()[0][size_class]
            rot = frustum_angle

            h, w, l, tx, ty, tz, ry = from_prediction_to_label_format(
                                            center, angle_class, angle_res, 
                                            size_class, size_res, rot)
            predictions_list.append([bbox_3D.pose.position.x, bbox_3D.pose.position.y, bbox_3D.pose.position.z, h, w, l, -ry])

        self.pub_pred_bbox(predictions_list, timestamp)


if __name__ == '__main__':
    print("Starting frustrum pointnet subscriber node...")

    pub = KittiSubscriber()
