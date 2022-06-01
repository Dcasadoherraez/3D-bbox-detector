import torch
import numpy as np

from sensor_msgs import point_cloud2
from std_msgs.msg import Header
import jsk_recognition_msgs.msg as jsk_msgs
from sensor_msgs.msg import PointField

from frustum_pointnet.models.frustum_pointnets import FrustumPointNetv1

darknet_to_pointnet = {'person': 'Pedestrian', 'car': 'Car'}
# CAMERA_FRAME = 'zed2i_left_camera_frame'
CAMERA_FRAME = 'fpointnet_frame'

# Parent class for all Frustum Pointnet nodes
class FPointnetNode:
    def __init__(self, model_root, n_classes) -> None:
        self.load_pretrained = model_root
        self.n_classes = n_classes

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


    def array2pc(self, points, timestamp):
        '''Convert numpy array in ros PointCloud2 msg'''
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]

        header = Header()
        header.frame_id = CAMERA_FRAME
        header.stamp = timestamp # rospy.Time.now()
        points = points.squeeze().reshape(-1, 3)

        msg = point_cloud2.create_cloud(header, fields, points)
        return msg

    def get_frustum_msg(self, frustums, timestamp):
        '''Publish lidar points in frustums'''
        if len(frustums) == 0:
            return

        frustum = np.vstack(frustums)
        lidar_window_msg = self.array2pc(frustum, timestamp)

        return lidar_window_msg

    def get_pc_msg(self, pointcloud, timestamp):
        '''Get pointcloud message'''
        pc_msg = self.array2pc(pointcloud, timestamp)
        return pc_msg

    def get_pred_bbox_msg(self, predictions, timestamp):
        '''Publish predicted object 3D bounding boxes'''

        bbox_array_3D = jsk_msgs.BoundingBoxArray()
        bbox_array_3D.boxes = []
        bbox_array_3D.header = Header()
        bbox_array_3D.header.stamp = timestamp
        bbox_array_3D.header.frame_id = CAMERA_FRAME

        i = 0
        while i < len(predictions):
            tx, ty, tz, h, w, l, ry = predictions[i]

            # 3D Bounding box
            bbox_3D = jsk_msgs.BoundingBox()
            bbox_3D.header.frame_id = CAMERA_FRAME
            bbox_3D.dimensions.x = l
            bbox_3D.dimensions.y = h
            bbox_3D.dimensions.z = w
            bbox_3D.pose.position.x = tx
            bbox_3D.pose.position.y = ty + h/2
            bbox_3D.pose.position.z = tz

            roll, pitch, yaw = 0, ry, 0
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

        return bbox_array_3D
