import numpy as np
from math import sin, cos, radians
from KittiDataset import KittiDataset
from KittiUtils import BBox3D, BBox2D

from mayavi import mlab

class KittiVisualizer:
    def __init__(self):
        self.figure = mlab.figure(bgcolor=(0,0,0), fgcolor=(1,1,1), size=(1280, 720))

    def visualize(self, pointcloud, calib, objects, labels_objects=None):
        """
            Visualize the Scene including Point Cloud & 3D Boxes 

            Args:
                pointcloud: numpy array (points_n, 3)
                calib: Kitti Calibration Object 
                objects: list of KittiObject 
        """
        # Point Cloud
        self.visuallize_pointcloud(pointcloud)
        self.calib = calib
        # 3D Boxes
        if labels_objects is not None:
            objects.extend(labels_objects)

        for obj in objects:
            bbox_3d = obj.bbox_3d
            score = obj.score
            color = self.get_box_color(obj.id)

            self.visualize_3d_bbox(bbox_3d, color)

            # draw score if not label
            if not obj.is_label:
                self.draw_text(*bbox_3d.pos, text=str(score), color=color)

    def visuallize_pointcloud(self, pointcloud):
        pointcloud = self.to_numpy(pointcloud)
        mlab.points3d(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], 
                    colormap='gnuplot', scale_factor=1, mode="point",  figure=self.figure)
        self.visualize_axes()

    def visualize_axes(self):
        l = 4 # axis_length
        w = 1
        mlab.plot3d([0, l], [0, 0], [0, 0], color=(0, 0, 1), line_width=w, figure=self.figure) # x
        mlab.plot3d([0, 0], [0, l], [0, 0], color=(0, 1, 0), line_width=w, figure=self.figure) # y
        mlab.plot3d([0, 0], [0, 0], [0, l], color=(1, 0, 0), line_width=w, figure=self.figure) # z

    def visualize_3d_bbox(self, bbox: BBox3D, color=(0,1,0)):
        corners = self.convert_3d_bbox_to_corners(bbox)
        self.visualize_box_corners(corners, color)

    def visualize_box_corners(self, corners, clr):
        if corners.shape[0] != 8:
            print("Invalid box format")
            return

        c0 = corners[0]
        c1 = corners[1] 
        c2 = corners[2] 
        c3 = corners[3] 
        c4 = corners[4] 
        c5 = corners[5] 
        c6 = corners[6] 
        c7 = corners[7] 

        # top suqare
        self.draw_line(c0, c1, clr)
        self.draw_line(c0, c2, clr)
        self.draw_line(c3, c1, clr)
        self.draw_line(c3, c2, clr)
        # bottom square
        self.draw_line(c4, c5, clr)
        self.draw_line(c4, c6, clr)
        self.draw_line(c7, c5, clr)
        self.draw_line(c7, c6, clr)
        # vertical edges
        self.draw_line(c0, c4, clr)
        self.draw_line(c1, c5, clr)
        self.draw_line(c2, c6, clr)
        self.draw_line(c3, c7, clr)

    def convert_3d_bbox_to_corners(self, bbox: BBox3D):
        """
            convert BBox3D with x,y,z, width, height, depth .. to 8 corners
                    h
              3 -------- 1
          w  /|         /|
            2 -------- 0 . d
            | |        | |
            . 7 -------- 5
            |/         |/
            6 -------- 4

                        z    x
                        |   / 
                        |  /
                        | /
                y--------/
        """
        x = bbox.x
        y = bbox.y
        z = bbox.z
        w = bbox.width  # y
        h = bbox.height # z
        l = bbox.length # x
        angle = bbox.rotation

        # convert x, y, z from rectified cam coordinates to velodyne coordinates
        point = np.array([x, y, z]).reshape(1,3)
        # point = self.calib.rectified_camera_to_velodyne(point)

        x = point[0,0]
        y = point[0,1]
        z = point[0,2]

        print(point)
        print(x,y,z)
        print("==========")

        top_corners = np.array([
            [x, y, z],
            [x+l, y, z],
            [x, y+w, z],
            [x+l, y+w, z]
        ])

        # same coordinates but z = z_top - box_height
        bottom_corners = top_corners - np.array([0,0, h])

        # concatinate 
        corners = np.concatenate((top_corners,bottom_corners), axis=0)

        print(corners.shape)

        # ======== Rotation ========          
        cosa = cos(angle)
        sina = sin(angle)

        # 3x3 Rotation Matrix along z 
        R = np.array([
            [ cosa, sina, 0],
            [-sina, cosa, 0],
            [ 0,    0,    1]
        ])

        # Translate the box to origin to perform rotation
        center = np.array([x+l/2, y+w/2, 0])
        centered_corners = corners - center

        # Rotate
        rotated_corners = np.dot( R, centered_corners.T ).T

        # Translate it back to its position
        corners = rotated_corners + center

        # output of sin & cos sometimes is e-17 instead of 0
        corners = np.round(corners, decimals=10)

        return corners

    def draw_line(self, corner1, corner2, clr):
        x = 0
        y = 1
        z = 2
        mlab.plot3d([corner1[x], corner2[x]], [corner1[y], corner2[y]], [corner1[z], corner2[z]],
                    line_width=2, color=clr, figure=self.figure)

    def draw_text(x, y, z, text, color):
        mlab.text3d(0,0,0, text, scale=0.3, color=color, figure=self.figure)
    
    def get_box_color(self, class_id):
        if type(class_id) == str:
            class_id = encode_classname(class_id)

        colors = [
            (0,1,0),
            (0,0,1),
            (0,1,1),
            (1,1,0),
            (0.6,0.8,0.2)
        ]

        return colors[class_id]

    def to_numpy(self, pointcloud):
        if not isinstance(pointcloud, np.ndarray):
            return pointcloud.cpu().numpy()
        return pointcloud


KITTI = KittiDataset()
_, pointcloud, labels, calib = KITTI[7]
print(pointcloud.shape)

visualizer = KittiVisualizer()
visualizer.visualize(pointcloud, calib, labels)


mlab.show(stop=True)


 



# # import open3d
# class Open3dVisualizer:
#     def visuallize_pcd(self, path):
#         pcd = open3d.io.read_point_cloud(path)
#         pcd_points = np.asarray(pcd.points)
#         print(pcd)
#         return pcd_points
#     def visuallize_pointcloud(self, pointcloud):
#         # convert numpy pointclod to open3d pointcloud
#         points = open3d.utility.Vector3dVector()
#         points.extend(pointcloud)
#         open3d_pointcloud = open3d.geometry.PointCloud(points)
#         print(open3d_pointcloud)
#         # visuallize
#         open3d.visualization.draw_geometries([open3d_pointcloud])
# ==================== Open3D ====================
# visualizer = Open3dVisualizer()
# p = visualizer.visuallize_pcd("/home/amrelsersy/KITTI/pcd/0000000000.pcd")
# visualizer.visuallize_pointcloud(p)
# PCD
# pcd = open3d.io.read_point_cloud("/home/amrelsersy/KITTI/pcd/0000000000.pcd")
# pcd_points = np.asarray(pcd.points)
# print(pcd_points.shape)
