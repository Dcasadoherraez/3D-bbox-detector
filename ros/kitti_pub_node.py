#!/usr/bin/env python
import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensor_msgs.msg import Image, PointCloud2, PointField
from vision_msgs.msg import BoundingBox2D, BoundingBox2DArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2
import kitti.kitti_util as utils
from std_msgs.msg import Header, String
import cv2
import rospy
import operator
import numpy as np


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


class KittiPublisher:
    def __init__(self) -> None:
        # Initialize all the paths
        self.dataset_root = DATASET_ROOT
        self.img_seq = IMG_SEQUENCE
        self.pc_seq = PC_SEQUENCE
        self.img_dataset_path = IMAGE_DATASET_PATH
        self.pc_dataset_path = PC_DATASET_PATH
        self.labels_path = LABELS_PATH

        rospy.init_node('kitti_publisher', anonymous=True)

        # CvBridge for OpenCV-ROS image transport
        self.bridge = CvBridge()

        # Messages that will be published by the node:
        
        # - RGB images
        self.img_pub = rospy.Publisher('image_publisher', Image, queue_size=1)
        
        # - Velodyne pointcloud
        self.pc_pub = rospy.Publisher(
            'pointcloud_publisher', PointCloud2, queue_size=1)
        
        # - Pointcloud frustrum from image size
        self.depth_pub = rospy.Publisher(
            'depth_publisher', PointCloud2, queue_size=1)

        # - 3D Annotated bounding boxes
        self.gt_bb_pub_3D = rospy.Publisher(
            'gt_bb_publisher_3D', BoundingBoxArray, queue_size=1)

        # - 2D Annotated bounding boxes
        self.gt_bb_pub_2D = rospy.Publisher(
            'gt_bb_publisher_2D', BoundingBox2DArray, queue_size=1)

        # Get all dataset paths
        self.images = sorted(os.listdir(self.img_dataset_path))
        self.pointclouds = sorted(os.listdir(self.pc_dataset_path))
        self.labels = sorted(utils.read_label(
            self.labels_path), key=operator.attrgetter('frame_num'))

        # Initialize calib from calibration file
        self.calib = utils.Calibration(SEQ_CALIB_PATH)

        self.rate = rospy.Rate(PUB_RATE)

        self.curr_frame = 0

    def get_calibration(self, idx):
        '''Get calibration file '''
        assert(idx < self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return utils.Calibration(calib_filename)

    def pub_img(self, image_name, timestamp):
        '''Publish image as ros msg'''
        image = cv2.imread(image_name)
        h, w, c = image.shape
        image_msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        image_msg.header.stamp = timestamp
        self.img_pub.publish(image_msg)

        return image

    def array2pc(self, points):
        '''Convert numpy array in ros PointCloud2 msg'''
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]

        header = Header()
        header.frame_id = "base_link"
        header.stamp = rospy.Time.now()

        return point_cloud2.create_cloud(header, fields, points)

    def pub_pc(self, pc, timestamp):
        '''Publish pointcloud'''
        points = np.fromfile(pc, dtype=np.float32).reshape(-1, 4)
        pc_msg = self.array2pc(points)
        pc_msg.header.stamp = timestamp
        self.pc_pub.publish(pc_msg)

        return points

    def get_lidar_in_image_fov(self, pointcloud, xmin, ymin, xmax, ymax,
                               clip_distance=2.0):
        ''' Filter lidar points, keep those in image FOV '''
        pc_velo = pointcloud[:, :3]
        pts_2d = self.calib.project_velo_to_image(pc_velo)
        fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
                   (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
        fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
        imgfov_pc_velo = pointcloud[fov_inds, :]

        return imgfov_pc_velo, pts_2d, fov_inds

    def pub_depth(self, image, pointcloud, timestamp):
        '''Publish lidar points in FOV'''
        h, w, _ = image.shape
        lidar_window, _, _ = self.get_lidar_in_image_fov(
            pointcloud, 0, 0, w, h)
        lidar_window_msg = self.array2pc(lidar_window)
        lidar_window_msg.header.stamp = timestamp
        self.depth_pub.publish(lidar_window_msg)

        return lidar_window

    def pub_gt_bbox(self, timestamp):
        '''Publish target object bounding boxes'''
      
        objects = self.labels

        i = self.curr_frame

        gt_bbox_array_3D = []
        bbox_array_3D = BoundingBoxArray()
        bbox_array_3D.header = Header()
        bbox_array_3D.header.stamp = timestamp
        bbox_array_3D.header.frame_id = "base_link"

        gt_bbox_array_2D = []
        bbox_array_2D = BoundingBox2DArray()
        bbox_array_2D.header = Header()
        bbox_array_2D.header.stamp = timestamp
        bbox_array_2D.header.frame_id = "base_link"

        while i < len(objects):
            if objects[i].frame_num == self.curr_frame and objects[i].type != 'DontCare':

                # 3D Bounding box
                obj = objects[i]
                bbox_3D = BoundingBox()
                bbox_3D.header.frame_id = "base_link"
                bbox_3D.dimensions.x = obj.w
                bbox_3D.dimensions.y = obj.l
                bbox_3D.dimensions.z = obj.h
                bbox_3D.pose.position.x = obj.t[2]
                bbox_3D.pose.position.y = -obj.t[0]
                bbox_3D.pose.position.z = -obj.t[1]/2

                roll, pitch, yaw = 0, 0, -obj.ry
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

                bbox_3D.label = obj.type
                bbox_3D.value = obj.class_id

                gt_bbox_array_3D.append(bbox_3D)

                # 2D Bounding box
                bbox_2D = BoundingBox2D()
                w = obj.xmin - obj.xmax
                h = obj.ymin - obj.ymax

                bbox_2D.center.x = obj.xmin + w //2
                bbox_2D.center.y = obj.ymin + h //2
                bbox_2D.center.theta = 0
                bbox_2D.size_x = w
                bbox_2D.size_y = h
                
                gt_bbox_array_2D.append(bbox_2D)

            elif objects[i].frame_num > self.curr_frame:
                break

            i += 1

        
        bbox_array_3D.boxes = gt_bbox_array_3D
        bbox_array_2D.boxes = gt_bbox_array_2D

        self.gt_bb_pub_3D.publish(bbox_array_3D)
        self.gt_bb_pub_2D.publish(bbox_array_2D)


    def pub_loop(self):
        '''Main node loop for topic publishing'''
        for image, pc in zip(self.images, self.pointclouds):
            if rospy.is_shutdown():
                break

            image_name = os.path.join(IMAGE_DATASET_PATH, image)
            pc_name = os.path.join(PC_DATASET_PATH, pc)

            timestamp = rospy.Time.now()
            image = self.pub_img(image_name, timestamp)
            pointcloud = self.pub_pc(pc_name, timestamp)
            self.pub_depth(image, pointcloud, timestamp)
            self.pub_gt_bbox(timestamp)

            self.curr_frame += 1
            self.rate.sleep()


if __name__ == '__main__':
    print("Starting image publisher node...")

    pub = KittiPublisher()
    try:
        pub.pub_loop()
    except rospy.ROSInterruptException:
        pass

'''

    def extract_frustum_data(self, idx_filename, idx, split, image, pointcloud, pointcloud_2d, fov_inds, perturb_box2d=False, augmentX=1, type_whitelist=['Car'], with_image=False):
        Extract point clouds and corresponding annotations in frustums
            defined generated from 2D bounding boxes
            Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

        Input:
            idx_filename: string, each line of the file is a sample ID
            split: string, either training or testing
            perturb_box2d: bool, whether to perturb the box2d
                (used for data augmentation in train set)
            augmentX: scalar, how many augmentations to have for each 2D box.
            type_whitelist: a list of strings, object types we are interested in.
        

        box2d_list = []  # [xmin,ymin,xmax,ymax]
        box3d_list = []  # (8,3) array in rect camera coord
        heading_list = []

        objects = self.labels

        pos_cnt = 0
        all_cnt = 0

        #print('------------- ', data_idx)
        calib = self.calib  # 3 by 4 matrix

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist:
                continue

            # 2D BOX: Get pts rect backprojected
            box2d = objects[obj_idx].box2d
            for _ in range(augmentX):
                xmin, ymin, xmax, ymax = box2d
                box_fov_inds = (pointcloud_2d[:, 0] < xmax) & \
                               (pointcloud_2d[:, 0] >= xmin) & \
                               (pointcloud_2d[:, 1] < ymax) & \
                               (pointcloud_2d[:, 1] >= ymin)
                box_fov_inds = box_fov_inds & fov_inds
                pc_in_box_fov = pointcloud[box_fov_inds, :]  # (1607, 4)

                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                uvdepth = np.zeros((1, 3))
                uvdepth[0, 0:2] = box2d_center
                uvdepth[0, 2] = 20  # some random depth
                box2d_center_rect = calib.project_image_to_rect(uvdepth)
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                                box2d_center_rect[0, 0])
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)  # (8, 2)(8, 3)
                _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)  # (375, 4)(1607,)
                label = np.zeros((pc_in_box_fov.shape[0]))  # (1607,)
                label[inds] = 1

                # Get 3D BOX heading
                heading_angle = obj.ry  # 0.01

                # Get 3D BOX size
                # array([1.2 , 0.48, 1.89])
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if ymax-ymin < 25 or np.sum(label) == 0:
                    continue

                box2d_list.append(box3d_pts_2d)
                box3d_list.append(box3d_pts_3d)
                heading_list.append(heading_angle)

                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]

        print('Average pos ratio: %f' % (pos_cnt/float(all_cnt)))
        print('Average npoints: %f' % (float(all_cnt)/len(box3d_list)))
        
        return box3d_list, box2d_list, heading_list

'''