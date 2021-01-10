import numpy as np

# ================================================
class BBox2D:
    def __init__(self, bbox):
        self.x = bbox[0]
        self.y = bbox[1]
        self.width = bbox[2]
        self.height = bbox[3]
        self.top_left = (self.x,self.y)

class BBox3D:
    def __init__(self, x, y, z, h, w, l, rotation):
        self.pos = (x,y,z)
        self.dims = (h,w,l)
        self.x = x
        self.y = y
        self.z = z
        self.height = h # z length
        self.width  = w # y length
        self.length = l # x length
        self.rotation = rotation
# ================================================

class KittiObject:
    def __init__(self, bbox_3d, class_name, label_dict, score=1, bbox_2d=None):
        self.bbox_3d = bbox_3d
        self.name = class_name
        self.score = score
        self.dict = label_dict
        self.bbox_2d = bbox_2d
        self.id = self.encode_classname(self.name)
        self.is_label = False

    def encode_classname(self, classname):
        class_to_id = {
            'Car': 0,
            'Van': 0,
            'Trunk': 0,
            'Pedestrian': 1,
            'Person_sitting': 1,
            'Cyclist': 2
        }

        return class_to_id[classname]

# ================================================
class KittiCalibration:
    """
        Perform different types of calibration between camera & LIDAR

        image = Projection * Camera3D_after_rectification
        image = Projection * R_Rectification * Camera3D_reference

    """
    def __init__(self, calib_dict):
        self.P0 = np.asarray(calib_dict["P0"])
        self.P1 = np.asarray(calib_dict["P1"])
        self.P3 = np.asarray(calib_dict["P3"])
        # Projection Matrix (Intrensic) .. from camera 3d (after rectification) to image coord.
        self.P2 = np.asarray(calib_dict["P2"]).reshape(3, 4)
        # rectification rotation matrix 3x3
        self.R0_rect = np.asarray(calib_dict["R0_rect"]).reshape(3,3)
        # Extrensic Transilation-Rotation Matrix from LIDAR to Cam ref(before rectification) 
        # Composed of 3x3 Rotation & a column of 3x1 Transilation
        self.Tr_velo_to_cam = np.asarray(calib_dict["Tr_velo_to_cam"]).reshape(3,4)

        self.Tr_cam_to_velo = self.inverse_Tr(self.Tr_velo_to_cam)

    def inverse_Tr(self, T):
        """ 
            get inverse of Transilation Rotation 4x4 Matrix
            Args:
                T: 4x4 Matrix
                    ([
                        [R(3x3) t],
                        [0 0 0  1]
                    ])
            Return:
                Inverse: 4x4 Matrix
                    ([
                        [R^-1   -R^1 * t],
                        [0 0 0         1]
                    ])                
        """
        R = T[0:3, 0:3]
        t = T[0:3, 3]
        print(R.shape, t.shape)
        R_inv = np.linalg.inv(R)
        t_inv = np.dot(-R_inv, t).reshape(3,1)
        T_inv = np.hstack((R_inv, t_inv))
        T_inv = np.vstack( (T_inv, np.array([0,0,0,1])) )
        return T_inv

    def rectified_camera_to_velodyne(self, points):
        """
            Converts 3D Box in Camera coordinates(after rectification) to 3D Velodyne coordinates
            Args: points
                numpy array (N, 3) in cam coord, N is points number 
            return: 
                numpy array (N, 3) in velo coord.
        """
        # # add 1 homogenious to all points .. (N, 4)
        # points = np.hstack(( points, np.ones((points.shape[0],1), dtype=np.float) ))
        # # convert T velo_cam_ref matrix to 4x4 by adding 0 0 0 1 vector @ bottom ... to take the inverse
        # T_homogenious = np.vstack( (self.Tr_velo_to_cam, np.array([0, 0, 0, 1])) )
        # # inverse of cam to rect === rect_to_cam
        # R_homoginous = np.linalg.inv(self.R0_rect)
        # ### convert R_homoginous to homogenious ... from 3x3 to 4x4
        # R_homoginous = np.hstack( (R_homoginous, np.zeros((R_homoginous.shape[0], 1), dtype=np.float)) )
        # R_homoginous = np.vstack( (R_homoginous, np.array([0, 0, 0, 1])))
        # # inverse of T velo_cam_ref === T_cam_velo
        # T_rectified_cam_to_velo = np.linalg.inv(np.dot(R_homoginous, T_homogenious).T)
        # points_3d_velodyne = np.dot(points, T_rectified_cam_to_velo)

        # from rectified cam to ref cam
        R_rect_inv = np.linalg.inv(self.R0_rect)
        points_ref =  np.dot(R_rect_inv, points.T) # 3xN
        # add homogenious 4xN
        points_ref = np.vstack((points_ref, np.ones((1, points_ref.shape[1]), dtype=np.float)))

        # velodyne = ref_to_velo * points_ref
        points_3d_velodyne = np.dot(self.Tr_cam_to_velo, points_ref)

        return points_3d_velodyne.T

