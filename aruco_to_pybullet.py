#!/usr/bin/env python

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import sys
import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R


def make_transform(t, rpy):
    """
    Create a 4x4 homogeneous transformation matrix.
    
    Args:
        t: translation vector [x, y, z] in meters
        rpy: [roll, pitch, yaw] in **radians** (or degrees if degrees=True)
    
    Returns:
        4x4 numpy array
    """
    # Convert RPY to rotation matrix
    rot = R.from_euler('xyz', rpy)  # use degrees=True if your angles are in degrees
    R_mat = rot.as_matrix()
    
    # Build homogeneous matrix
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

def import_object(path, obj_pose):
    '''
    Import objects in simulation.
    Args:
        path (str): path of the object file
        obj_pose (list): pose of the object as [position, quaternions]
    '''

    object_shape = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                        fileName=path,
                                        meshScale=[1, 1, 1])
    object_visual_shape = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                fileName=path,
                                                meshScale=[1, 1, 1]) 
    object_id = p.createMultiBody(baseMass=0.05,              
                                        baseCollisionShapeIndex=object_shape,
                                        baseVisualShapeIndex=object_visual_shape,
                                        basePosition=obj_pose[0],
                                        baseOrientation=obj_pose[1])
	
    return object_id

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	default="DICT_5X5_100",
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

camera_matrix = np.array([
    [910.1155107777962, 0.0, 360.3277519024787],
    [0.0, 910.2233367566544, 372.6634999577232],
    [0.0, 0.0, 1.0]
])

dist_coeffs = np.array([
    0.0212284835698144,
    0.8546829039917951,
    0.0034281408326615323,
    0.0005749116561059772,
    -3.217248182814475
])

# Marker side length in meters (change this to your actual printed marker size)
marker_length = 0.053  # 4 cm

# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(
		args["type"]))
	sys.exit(0)

# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()

# Start PyBullet simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # for plane.urdf
p.setGravity(0, 0, -9.81)

# Load ground plane
plane_id = p.loadURDF("plane.urdf")
object_id = import_object("/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/aruco2pybullet/milk_MD.stl", ([0.5, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0])))


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=6).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

	# detect ArUco markers in the input frame
	(corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
		arucoDict, parameters=arucoParams)

	# verify *at least* one ArUco marker was detected
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()

		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned
			# in top-left, top-right, bottom-right, and bottom-left
			# order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
				[markerCorner], marker_length, camera_matrix, dist_coeffs
			)

			# tvecs contains the 3D position of the marker w.r.t the camera
			#tvec = tvecs[0][0] - [0.1, 0, 0] #! PARA ADAM
			tvec = tvecs[0][0]   # Adjust for marker center offset
			rvec = rvecs[0][0]
			print(f"Marker {markerID} position (x,y,z) in meters: {tvec}")

			# Convert rvec to rotation matrix
			R_camera, _ = cv2.Rodrigues(rvec)

			# Build homogeneous transform T_marker_in_camera
			T_marker_in_camera = np.eye(4)
			T_marker_in_camera[:3, :3] = R_camera
			T_marker_in_camera[:3, 3] = tvec

			# Define fixed transforms based for ADAM
			""" T_base_shoulder = make_transform([0.007, 0, 1.254], [0, 0, 0])
			T_shoulder_joint = make_transform([0.115, 0, 0.030], [0, np.pi/4, 0])
			T_joint_support = make_transform([0.027, 0, 0.0425], [0, 0, 0])
			T_support_camera = make_transform([0.01803, 0.0175, 0], [-np.pi/2, 0, -np.pi/2]) """
			T_marker_in_robot = T_marker_in_camera.copy()

			#T_marker_in_robot = T_base_shoulder @ T_shoulder_joint @ T_joint_support @ T_support_camera @ T_marker_in_camera
			# Rotación de -90° en Y y 180° en Z
			R_y = R.from_euler('y', -90, degrees=True).as_matrix()
			R_z = R.from_euler('z', 180, degrees=True).as_matrix()

			# Aplicamos la rotación: primero Y, luego Z
			T_marker_in_robot[:3, :3] = R_z @ R_y @ T_marker_in_camera[:3, :3]

			# Extract position & orientation
			pos = T_marker_in_robot[:3, 3]
			rot = R.from_matrix(T_marker_in_robot[:3, :3])
			quat = rot.as_quat()  # (x, y, z, w)
			quat = np.round(quat, 2)

			print("Marker position in robot base:", pos)
			print("Marker orientation in robot base (quat):", quat)

			# Update object in PyBullet
			p.resetBasePositionAndOrientation(object_id, pos, quat)
			p.stepSimulation()

			cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length)

			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			# draw the bounding box of the ArUCo detection
			cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

			# compute and draw the center (x, y)-coordinates of the
			# ArUco marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

			# draw the ArUco marker ID on the frame
			cv2.putText(frame, str(markerID),
				(topLeft[0], topLeft[1] - 15),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
p.disconnect()
