#!/usr/bin/env python3
# ADAM_Aruco_Pathwise_withOrientPath.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import threading
import gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# GPflow / ProGP imports
import tensorflow as tf
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import updates
from ProGP import ProGpMp

# PyBullet
import pybullet as p
import pybullet_data

# Camera / ArUco
from imutils.video import VideoStream
import imutils
import cv2
from scipy.spatial.transform import Rotation as R

#ROS
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Intento importar ADAM (opcional)
ADAM_AVAILABLE = False
try:
    from Adam_sim.scripts.adam import ADAM
    ADAM_AVAILABLE = True
except Exception as e:
    print("Warning: ADAM import failed:", e)
    ADAM_AVAILABLE = False

# ---------- Paths / Config ----------
CSV_FOLDER = "/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/ExpCartesianADAM"
ADAM_URDF_PATH = "/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/Adam_sim/models/robot/rb1_base_description/robots/robotDummy.urdf"

# ---------- Globals ----------
last_changed = None
last_pts = None          # Nx3 mm
last_quats = None        # Nx4 (x,y,z,w)
marker_map = None
all_markers_detected = None

mean_gmp_arr = np.array([])   # (3, Nt) mm prior
mean_quat_arr = None          # (4, Nt) prior (si existe)
vias = np.array([])
via_t = np.array([])
last_final_line, prior_line = None, None
ax = None
pb_global = None
test_x = None

camera_last_quat = None
camera_last_pos_mm = None
camera_tracked_idx = 0
frame_artists = []
arrow_len = 1.0  # mm length for quiver axes
first_time = True

# --------------------------
# PyBulletManager (soporta ADAM si está disponible)
# --------------------------
class PyBulletManager:
    def __init__(self, start_adam=False, adam_urdf=None, use_gui=True, gravity=-9.81, time_step=1./240., wait_adam_timeout=8.0):
        self.client = None
        self.time_step = time_step
        self._sim_thread = None
        self._sim_running = False
        self.bodies = []
        self.lock = threading.Lock()
        self.path_ids = []
        self.path_m = None
        self.path_version = 0

        # NEW: store orientation path (quaternions) aligned with path_m
        self.path_quat = None       # shape (N,4) in [x,y,z,w] order, in same discretization as path_m
        self.path_quat_version = 0

        # ADAM related
        self.adam = None
        self.adam_ready = False
        self.adam_exc = None
        self._adam_thread = None
        self._start_adam = start_adam and ADAM_AVAILABLE
        self._adam_urdf = adam_urdf if adam_urdf is not None else ADAM_URDF_PATH
        self.wait_adam_timeout = wait_adam_timeout
        
        #Variables ROS
        # Create a publisher for the robot's pose
        # Initialize the ROS node
        rospy.init_node('adam_ros_node', anonymous=True)
        self.arm_joint_pub = {'left': rospy.Publisher('/robot/left_arm/scaled_pos_traj_controller/command', JointTrajectory, queue_size=1),
                            'right': rospy.Publisher('/robot/right_arm/scaled_pos_traj_controller/command', JointTrajectory, queue_size=1)}

        if self._start_adam:
            self._adam_thread = threading.Thread(target=self._init_adam, daemon=True)
            self._adam_thread.start()
            waited = 0.0
            poll = 0.15
            while waited < self.wait_adam_timeout:
                if self.adam_ready or self.adam_exc is not None:
                    break
                time.sleep(poll)
                waited += poll

        # choose client
        cid = None
        if self._start_adam and self.adam_ready:
            cid = self._extract_adam_client()
            if cid is not None:
                print("PyBulletManager: usando physicsClientId expuesto por ADAM:", cid)
            else:
                print("PyBulletManager: ADAM arrancado pero no expone client id -> creamos cliente propio GUI.")
        flags = p.GUI if use_gui else p.DIRECT
        if cid is None:
            self.client = p.connect(flags)
        else:
            # if we extracted ADAM client id, use it
            self.client = cid

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0,0,gravity, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)
        try:
            _ = p.loadURDF("plane.urdf", physicsClientId=self.client)
        except Exception:
            pass

        # start stepping thread
        self._sim_running = True
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

    def _init_adam(self):
        try:
            print("PyBulletManager: arrancando ADAM (background)...")
            self.adam = ADAM(self._adam_urdf, useRealTimeSimulation=True, used_fixed_base=True, use_ros=False)
            self.adam_ready = True
            print("PyBulletManager: ADAM iniciado.")
        except Exception as e:
            self.adam_exc = e
            self.adam_ready = False
            print("PyBulletManager: fallo arrancando ADAM:", e)

    def _extract_adam_client(self):
        if self.adam is None:
            return None
        cand_names = ['physicsClientId', 'physics_client', 'client', 'pybullet_client', 'pb_client',
                    '_physics_client', 'physicsClient', 'client_id', '_client', 'pclient']
        for name in cand_names:
            if hasattr(self.adam, name):
                v = getattr(self.adam, name)
                if isinstance(v, int) and v >= 0:
                    return v
                if hasattr(v, 'client'):
                    c = getattr(v, 'client')
                    if isinstance(c, int) and c >= 0:
                        return c
        # fallback: inspect __dict__
        for _, val in getattr(self.adam, '__dict__', {}).items():
            if isinstance(val, int) and val >= 0 and val < 100:
                return val
        return None

    def _sim_loop(self):
        while self._sim_running:
            with self.lock:
                try:
                    p.stepSimulation(physicsClientId=self.client)
                except Exception:
                    pass
            time.sleep(self.time_step)

    def create_spheres(self, positions_cm, radius_cm=1.0, rgba=[1,0,0,1]):
        with self.lock:
            for b in self.bodies:
                try: p.removeBody(b, physicsClientId=self.client)
                except Exception: pass
            self.bodies = []
            positions_m = np.array(positions_cm)/100.0
            radius_m = float(radius_cm)/100.0
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius_m, rgbaColor=rgba, physicsClientId=self.client)
            for pos in positions_m:
                try:
                    b = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[float(pos[0]),float(pos[1]),float(pos[2])], physicsClientId=self.client)
                    self.bodies.append(b)
                except Exception:
                    pass
        return self.bodies

    def update_spheres(self, positions_cm):
        with self.lock:
            if not self.bodies: return
            positions_m = np.array(positions_cm)/100.0
            if positions_m.ndim == 1 and positions_m.size == 3:
                positions_m = positions_m.reshape(1,3)
            if positions_m.ndim == 2:
                n = positions_m.shape[0]
                m = min(n, len(self.bodies))
                for bid, pos in zip(self.bodies[:m], positions_m[:m]):
                    try:
                        p.resetBasePositionAndOrientation(bid, [float(pos[0]),float(pos[1]),float(pos[2])], [0,0,0,1], physicsClientId=self.client)
                    except Exception:
                        pass

    def set_path(self, path_cm):
        """Guardar path de posiciones (cm -> internamente lo pasamos a m)."""
        with self.lock:
            if path_cm is None:
                self.path_m = None
            else:
                arr = np.array(path_cm, dtype=float)
                self.path_m = arr * 0.01
            self.path_version += 1

    def set_orientations(self, path_quat):
        """
        Guardar path de orientaciones (esperamos array (N,4) con quaternions [x,y,z,w]).
        Debe estar alineado en longitud con path_m idealmente.
        """
        with self.lock:
            if path_quat is None:
                self.path_quat = None
            else:
                arr = np.array(path_quat, dtype=float)
                # forzar shape (-1,4)
                if arr.ndim == 1 and arr.size == 4:
                    arr = arr.reshape(1,4)
                if arr.ndim == 2 and arr.shape[1] == 4:
                    # normalizar quaternions, evitar ceros
                    norms = np.linalg.norm(arr, axis=1)
                    for i,n in enumerate(norms):
                        if n < 1e-8:
                            arr[i] = np.array([0.,0.,0.,1.])
                        else:
                            arr[i] = arr[i] / n
                    self.path_quat = arr.copy()
                else:
                    # formato inválido -> ignorar
                    self.path_quat = None
            self.path_quat_version += 1

    def draw_path(self, points_cm, step=5, color=[1,0,0], width=2.0, lifeTime=0):
        with self.lock:
            for pid in self.path_ids:
                try:
                    p.removeUserDebugItem(pid, physicsClientId=self.client)
                except Exception:
                    pass
            self.path_ids = []
            if points_cm is None: return
            pts = np.array(points_cm, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3: return
            pts_m = pts * 0.01
            for i in range(0, pts.shape[0]-1, step):
                a = pts_m[i].tolist()
                b = pts_m[min(i+step, pts.shape[0]-1)].tolist()
                try:
                    pid = p.addUserDebugLine(a, b, lineColorRGB=color, lineWidth=width, lifeTime=lifeTime, physicsClientId=self.client)
                    self.path_ids.append(pid)
                except Exception:
                    pass

    def start_adam_follow(self, speed=1.0, ee_side='right', ee_link_name='hand'):
        self._follow_speed = float(speed)
        self._initialized = True
        self._ee_side = ee_side
        self._ee_link_name = ee_link_name
        self._robot_running = True
        self._robot_thread = threading.Thread(target=self._adam_robot_loop, daemon=True)
        self._robot_thread.start()

    def stop_adam_follow(self):
        self._robot_running = False
        if self._robot_thread is not None:
            self._robot_thread.join(timeout=1.0)
            self._robot_thread = None
            
    def convert_joints_to_msg(self, arm, joint_positions):
        """
        Crea un mensaje JointTrajectory a partir de una lista de nombres y posiciones.
        
        Args:
            joint_names (list of str): Nombres de las articulaciones.
            joint_positions (list of float): Posiciones deseadas para cada articulación.
            
        Returns:
            JointTrajectory: Mensaje listo para publicar.
        """
        traj_msg = JointTrajectory()
        traj_msg.joint_names = [f'robot_{arm}_arm_elbow_joint', f'robot_{arm}_arm_shoulder_lift_joint', f'robot_{arm}_arm_shoulder_pan_joint',
                                f'robot_{arm}_arm_wrist_1_joint', f'robot_{arm}_arm_wrist_2_joint', f'robot_{arm}_arm_wrist_3_joint']
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0, 0, 0, 0, 0, 0]
        point.accelerations = [0, 0, 0, 0, 0, 0]
        point.effort = [0.8682, 1.612, -1.5094, 0.3217, -0.2149, 0.0096]
        point.time_from_start.secs = 1  # Tiempo opcional para alcanzar la posición
        
        traj_msg.points.append(point)

        return traj_msg

    def _adam_robot_loop(self):
        """
        Bucle que sigue self.path_m y usa self.path_quat (si existe) para orientar el efector.
        Ambas estructuras deben estar alineadas en longitud idealmente; si no, hacemos un mapeo por índice.
        """
        global last_quats

        # quaternion fallback (x,y,z,w)
        #default_quat = [0.09417623281478882, 0.7259091734886169, -0.05370301008224487, 0.6791926622390747]
        # intentar usar last_quats si existe al inicializar
        try:
            if last_quats is not None and getattr(last_quats, 'shape', None) is not None and last_quats.shape[0] > 0:
                default_quat = last_quats[0].tolist()
        except Exception:
            pass

        # si ADAM puede dar la pose inicial del efector, úsala como fallback primario (sin forzar siempre)
        try:
            if getattr(self, 'adam', None) is not None:
                cur = self.adam.arm_kinematics.get_arm_link_pose(self._ee_side, target_link=self._ee_link_name)
                if cur is not None and len(cur) >= 2 and cur[1] is not None:
                    cq = np.array(cur[1], dtype=float).reshape(-1)
                    if cq.size == 4 and np.all(np.isfinite(cq)) and np.linalg.norm(cq) > 1e-8:
                        default_quat = (cq / np.linalg.norm(cq)).tolist()
        except Exception:
            pass

        while getattr(self, '_robot_running', False):
            self.adam.hand_kinematics.move_hand_to_dofs('left', [1000, 1000, 1000, 1000, 1000, 1000])
            with self.lock:
                local_path = None if self.path_m is None else (self.path_m.copy())
                local_version = self.path_version
                local_quats = None if self.path_quat is None else (None if self.path_quat is None else (self.path_quat.copy()))
                local_quat_version = self.path_quat_version

            if local_path is None or local_path.size == 0:
                # step ADAM if available
                try:
                    if getattr(self, 'adam', None) is not None:
                        self.adam.step()
                except Exception:
                    pass
                time.sleep(0.02)
                continue

            if self._initialized:
                self._initialized = False
                index = 0
            """ else:
                index = 0 """

            L = local_path.shape[0] if (local_path is not None) else 1
            # length of quaternion path if available
            QL = local_quats.shape[0] if (local_quats is not None) else 0

            while index < L and getattr(self, '_robot_running', False):
                with self.lock:
                    if local_version != self.path_version:
                        break
                    path_copy = self.path_m.copy() if self.path_m is not None else None
                    quat_copy = self.path_quat.copy() if self.path_quat is not None else None
                    local_quat_len = quat_copy.shape[0] if (quat_copy is not None) else 0

                if path_copy is None or path_copy.size == 0:
                    break

                target_pos = path_copy[index].tolist()   # meters

                # choose orientation: prefer path_quat if available and aligned, else try to map index -> mean_quat, else fallback
                q = default_quat
                try:
                    # if explicit per-step orientation path available and lengths match or at least not zero
                    if quat_copy is not None and local_quat_len > 0:
                        # if same length as path_copy, take same index; else map linearly
                        if local_quat_len == path_copy.shape[0]:
                            qcand = quat_copy[index]
                        else:
                            # map index -> idx_q (clamp)
                            idx_q = int(round(float(index) * float(local_quat_len - 1) / max(1.0, float(path_copy.shape[0] - 1))))
                            idx_q = max(0, min(local_quat_len - 1, idx_q))
                            qcand = quat_copy[idx_q]
                        qcand = np.asarray(qcand, dtype=float).reshape(-1)
                        if qcand.size == 4 and np.all(np.isfinite(qcand)) and np.linalg.norm(qcand) > 1e-8:
                            qcand = qcand / np.linalg.norm(qcand)
                            q = qcand.tolist()
                        else:
                            raise ValueError("qcand invalid")
                    else:
                        # fallback: try to get orientation from ADAM current pose (once)
                        try:
                            if getattr(self, 'adam', None) is not None:
                                cur = self.adam.arm_kinematics.get_arm_link_pose(self._ee_side, target_link=self._ee_link_name)
                                if cur is not None and len(cur) >= 2 and cur[1] is not None:
                                    cq = np.array(cur[1], dtype=float).reshape(-1)
                                    if cq.size == 4 and np.all(np.isfinite(cq)) and np.linalg.norm(cq) > 1e-8:
                                        q = (cq / np.linalg.norm(cq)).tolist()
                        except Exception:
                            pass
                except Exception:
                    # leave q as default_quat
                    q = default_quat
                #!PARA La DEMo;LUEGO QUIATRRRRRRR:
                q = default_quat
                # apply IK using q
                try:
                    ee_index = self.adam.arm_kinematics.get_arm_link_index(self._ee_side, self._ee_link_name)
                    joint_indices, rev_joint_indices = self.adam.arm_kinematics.get_arm_joint_indices(self._ee_side)
                    ik_solution = p.calculateInverseKinematics(self.adam.robot_id, ee_index, target_pos, q,
                                                    solver=0,
                                                    maxNumIterations=1000,
                                                    residualThreshold=.01)
                    arm_solution = [ik_solution[i] for i in rev_joint_indices]
                    arm_solution = self.adam.arm_kinematics.compute_closest_joints(self._ee_side, arm_solution)
                    for i, joint_id in enumerate(joint_indices):
                        p.setJointMotorControl2(self.adam.robot_id, joint_id, p.POSITION_CONTROL, arm_solution[i], physicsClientId=self.client)
                    self.adam.step()
                    #self.adam.ros.arm_publish_joint_trajectory(arm='left',joint_angles=arm_solution)
                    # Create a JointTrajectory message
                    #print("Arm solution",arm_solution)
                    aux1= arm_solution[0]
                    aux2 = arm_solution[2]
                    arm_solution[0] = aux2
                    arm_solution[2] = aux1
                    traj_msg = self.convert_joints_to_msg('left', arm_solution)
                    #print("Traj msg", traj_msg)
                    # Publish the message
                    self.arm_joint_pub['left'].publish(traj_msg)
                except Exception:
                    # fallback: just step
                    try:
                        self.adam.step()
                    except Exception:
                        pass

                # pacing / wait
                self.adam.wait(0.3)
                index += 1

            time.sleep(0.005)
            #self.stop_adam_follow()

    def stop(self, disconnect_client=False):
        self.stop_adam_follow()
        self._sim_running = False
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=1.0)
            self._sim_thread = None
        with self.lock:
            for b in self.bodies:
                try: p.removeBody(b, physicsClientId=self.client)
                except Exception: pass
            self.bodies = []
        if disconnect_client:
            try:
                p.disconnect(self.client)
            except Exception:
                pass

# --------------------------
# Camera-driven GUI (similar a CameraOnlyPoints3D del original)
# --------------------------
class CameraOnlyPoints3D:
    def __init__(self, points_mm, on_update=None, fig_size=(8,8)):
        self.points = np.array(points_mm, float)
        self.last_points = self.points.copy()
        self.on_update = on_update

        self.fig = plt.figure(figsize=fig_size)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.mouse_init(rotate_btn=3, zoom_btn=2)

        xs, ys, zs = self.points.T
        self.scatter = self.ax.scatter(xs, ys, zs, s=100, c='red')

        self.texts = []
        for i, (x, y, z) in enumerate(self.points):
            txt = self.ax.text(x, y, z, str(i+1),
                            color='white', ha='center', va='center',
                            bbox=dict(boxstyle='circle', fc='blue', alpha=0.6))
            self.texts.append(txt)

        self.ax.set_title('Camera-controlled via-points (mm)')
        self._cam_thread = None
        self._cam_running = False
        self._last_pose_mm = None
        self._last_quat = None

    def _notify(self):
        diffs = np.any(self.points != self.last_points, axis=1)
        changed = np.where(diffs)[0].tolist()
        idxs = changed if changed else None
        if self.on_update:
            self.on_update(self.points.copy(), idxs)
        self.last_points[:] = self.points

    def get_positions(self):
        return self.points.copy()

    def show(self):
        plt.show()

    def update_point_mm(self, idx, new_point_mm, quat=None, set_text=True):
        global camera_last_quat, camera_last_pos_mm, camera_tracked_idx, pb_global, last_quats
        try:
            self.points[idx] = new_point_mm
            xs, ys, zs = self.points.T
            try:
                self.scatter._offsets3d = (xs, ys, zs)
            except Exception:
                self.scatter.set_offsets(np.c_[xs, ys])
                self.scatter.set_3d_properties(zs, zdir='z')
            if set_text:
                self.texts[idx].set_position((new_point_mm[0]/100, new_point_mm[1]/100))
                self.texts[idx].set_3d_properties(new_point_mm[2]/100, zdir='z')

            self._last_pose_mm = new_point_mm.copy()
            if quat is not None:
                self._last_quat = quat
                camera_last_quat = quat
                if last_quats is not None:
                    try:
                        last_quats[idx] = quat
                    except Exception:
                        pass

            camera_last_pos_mm = new_point_mm.copy()
            camera_tracked_idx = idx

            # PyBullet update by index (pb_global.update_spheres accepts dict in cm)
            if pb_global is not None:
                try:
                    pb_global.update_spheres({int(idx): (new_point_mm / 10.0).tolist()})
                    with pb_global.lock:
                        if 0 <= idx < len(pb_global.bodies):
                            pos_m = (new_point_mm / 1000.0).tolist()
                            q = quat.tolist() if quat is not None else [0,0,0,1]
                            try:
                                p.resetBasePositionAndOrientation(pb_global.bodies[idx], pos_m, q, physicsClientId=pb_global.client)
                            except Exception:
                                pass
                except Exception:
                    pass

            self._notify()
            self.fig.canvas.draw_idle()
        except Exception as e:
            print("update_point_mm error:", e)

    def start_camera_tracking(self,
                            vs_src=6,
                            aruco_type=cv2.aruco.DICT_5X5_100,
                            camera_matrix=None,
                            dist_coeffs=None,
                            marker_length_m=0.053,
                            tracked_idx=0,
                            apply_fixed_rotations=True,
                            video_w=1000):
        if camera_matrix is None or dist_coeffs is None:
            raise ValueError("Se requieren camera_matrix y dist_coeffs para start_camera_tracking()")

        self._cam_camMat = camera_matrix
        self._cam_dist = dist_coeffs
        self._cam_marker_len = marker_length_m
        self._cam_tracked_idx = tracked_idx
        self._cam_aruco_type = aruco_type
        self._cam_vs_src = vs_src
        self._cam_video_w = video_w
        self._cam_apply_fixrot = apply_fixed_rotations

        if self._cam_running:
            return
        self._cam_running = True

        def _cam_loop():
            arucoDict = cv2.aruco.getPredefinedDictionary(self._cam_aruco_type)
            if hasattr(cv2.aruco, 'DetectorParameters_create'):
                arucoParams = cv2.aruco.DetectorParameters_create()
            else:
                arucoParams = cv2.aruco.DetectorParameters()

            use_vs = False
            vs = None
            cap = None
            try:
                vs = VideoStream(src=self._cam_vs_src).start()
                time.sleep(1.0)
                use_vs = True
            except Exception:
                cap = cv2.VideoCapture(self._cam_vs_src)
                time.sleep(0.5)
                use_vs = False

            default_marker_length_m = None
            if isinstance(self._cam_marker_len, dict):
                default_marker_length_m = self._cam_marker_len.get("default", None)
            else:
                try:
                    default_marker_length_m = float(self._cam_marker_len)
                except Exception:
                    default_marker_length_m = None

            while self._cam_running:
                if use_vs:
                    frame = vs.read()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        frame = None

                if frame is None:
                    time.sleep(0.01)
                    continue
                frame = imutils.resize(frame, width=self._cam_video_w)

                try:
                    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
                except Exception:
                    corners, ids, rejected = [], None, None

                if ids is not None and len(corners) > 0:
                    try:
                        ids_flat = ids.flatten()
                    except Exception:
                        ids_flat = np.array(ids).reshape(-1)

                    for (markerCorner, markerID) in zip(corners, ids_flat):
                        try:
                            corners_pts = markerCorner.reshape((4, 2))
                            (topLeft, topRight, bottomRight, bottomLeft) = corners_pts

                            # choose marker length
                            if isinstance(self._cam_marker_len, dict):
                                marker_len_m = self._cam_marker_len.get(int(markerID), default_marker_length_m)
                            else:
                                marker_len_m = default_marker_length_m

                            if marker_len_m is None:
                                continue

                            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                                [markerCorner], marker_len_m, self._cam_camMat, self._cam_dist
                            )
                            tvec = tvecs[0][0]
                            rvec = rvecs[0][0]
                            R_camera, _ = cv2.Rodrigues(rvec)
                            T_marker_in_camera = np.eye(4)
                            T_marker_in_camera[:3, :3] = R_camera
                            T_marker_in_camera[:3, 3] = tvec

                            if self._cam_apply_fixrot:
                                T_base_shoulder = make_transform([0.007, 0, 1.254], [0, 0, 0])
                                T_shoulder_joint = make_transform([0.115, 0, 0.030], [0, np.pi/4, 0])
                                T_joint_support = make_transform([0.027, 0, 0.0425], [0, 0, 0])
                                T_support_camera = make_transform([0.01803, 0.0175, 0], [-np.pi/2, 0, -np.pi/2])
                                T_extra = make_transform([0, 0, 0], [0, np.pi/2, 0])
                                T_extra2 = make_transform([0, 0, 0], [0, 0, np.pi/2])
                                T_marker_in_robot = T_base_shoulder @ T_shoulder_joint @ T_joint_support @ T_support_camera @ T_marker_in_camera @ T_extra @T_extra2
                            else:
                                R_y = R.from_euler('y', -90, degrees=True).as_matrix()
                                R_z = R.from_euler('z', 180, degrees=True).as_matrix()
                                T_marker_in_robot = T_marker_in_camera.copy()
                                T_marker_in_robot[:3, :3] = R_z @ R_y @ T_marker_in_camera[:3, :3]

                            pos_m = T_marker_in_robot[:3, 3]
                            pos_mm = pos_m * 1000.0
                            rot_mat = T_marker_in_robot[:3, :3]
                            quat = R.from_matrix(rot_mat).as_quat()  # x,y,z,w
                            #self.adam.utils.draw_frame([pos_mm,quat], axis_length=0.1, line_width=4)

                            dest_idx = None
                            idx_map = self._cam_tracked_idx
                            if isinstance(idx_map, dict):
                                dest_idx = idx_map.get(int(markerID), None)
                            else:
                                try:
                                    base = int(idx_map)
                                    try:
                                        k = list(ids_flat).index(int(markerID))
                                    except Exception:
                                        k = 0
                                    dest_idx = base + k
                                except Exception:
                                    try:
                                        mid = int(markerID)
                                        if 0 <= mid < self.points.shape[0]:
                                            dest_idx = mid
                                    except Exception:
                                        dest_idx = None

                            if dest_idx is None or not (0 <= dest_idx < self.points.shape[0]):
                                continue

                            self.update_point_mm(dest_idx, pos_mm, quat=quat, set_text=True)

                            try:
                                cv2.drawFrameAxes(frame, self._cam_camMat, self._cam_dist, rvec, tvec, marker_len_m)
                            except Exception:
                                pass

                            # debug visuals
                            try:
                                topRight = (int(topRight[0]), int(topRight[1]))
                                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                                topLeft = (int(topLeft[0]), int(topLeft[1]))
                            except Exception:
                                pass

                            cv2.putText(frame, f"ID:{int(markerID)}->pt:{dest_idx}",
                                        (int(topLeft[0]), int(topLeft[1]) - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        except Exception:
                            continue

                try:
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self._cam_running = False
                        break
                except Exception:
                    pass

                time.sleep(0.001)

            try:
                if use_vs and vs is not None:
                    vs.stop()
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                cv2.destroyWindow("Frame")
            except Exception:
                pass

        self._cam_thread = threading.Thread(target=_cam_loop, daemon=True)
        self._cam_thread.start()

    def stop_camera(self):
        self._cam_running = False
        if self._cam_thread is not None:
            self._cam_thread.join(timeout=1.0)
            self._cam_thread = None

# --------------------------
# Helpers / pathwise update
# --------------------------
def make_transform(t, rpy):
    rot = R.from_euler('xyz', rpy)
    R_mat = rot.as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

def print_update_3d(pts, changed):
    global last_changed, last_pts, pb_global, camera_last_pos_mm, camera_last_quat
    last_changed = changed
    last_pts = pts.copy()

    if pb_global is not None and changed is not None:
        try:
            upd = {}
            for i in changed:
                all_markers_detected[i] = True
                upd[int(i)] = (pts[int(i)] / 10.0).tolist()  # mm->cm
            if upd:
                pb_global.update_spheres(upd)
        except Exception:
            pass

    # if camera provided an orientation for tracked index, apply to pb body
    if pb_global is not None and camera_last_pos_mm is not None and camera_last_quat is not None:
        try:
            idx = int(camera_tracked_idx)
            if 0 <= idx < len(pb_global.bodies):
                pos_m = (camera_last_pos_mm / 1000.0).tolist()
                q = camera_last_quat.tolist()
                with pb_global.lock:
                    p.resetBasePositionAndOrientation(pb_global.bodies[idx], pos_m, q, physicsClientId=pb_global.client)
        except Exception:
            pass

    if last_changed is not None:
        pathwise_update_fn()

def pathwise_update_fn():
    """
    Usa last_pts (mm) y last_quats (si existen) para condicionar el prior (mean_gmp_arr, mean_quat_arr)
    Actualiza también el path de orientaciones en pb_global llamando a set_orientations(...)
    """
    global mean_gmp_arr, mean_quat_arr, vias, via_t, last_changed, last_pts, last_quats, ax, pb_global, test_x, frame_artists, last_final_line, prior_line, first_time

    # clear previous frame artists
    try:
        for art in frame_artists:
            try:
                art.remove()
            except Exception:
                try:
                    if art in ax.collections:
                        ax.collections.remove(art)
                except Exception:
                    pass
        frame_artists[:] = []
    except Exception:
        frame_artists[:] = []

    # build kernels
    BaseClasses = (kernels.SquaredExponential,)*3
    base_kernels = [cls(lengthscales=1.0) for cls in BaseClasses]
    kernel = kernels.SeparateIndependent(base_kernels)

    # quaternion kernel if needed
    if mean_quat_arr is not None:
        BaseClassesQ = (kernels.SquaredExponential,)*4
        base_kernels_q = [cls(lengthscales=1.0) for cls in BaseClassesQ]
        kernel_q = kernels.SeparateIndependent(base_kernels_q)
    else:
        kernel_q = None

    via_t_local = np.array([via_t[0], via_t[1], via_t[-1]])
    X_obs = tf.convert_to_tensor(via_t_local, dtype=default_float())

    """ pts_copy = last_pts.copy()
    print("last_pts", last_pts)
    try:
        if first_time == True:
            first_time = False
            for i in range(len(marker_map)):
                pts_copy[i] = pts_copy[i] / 100.0
        else:
            for i in range(min(len(pts_copy), len(marker_map) if marker_map is not None else len(pts_copy))):
                pts_copy[i] = pts_copy[i] / 100.0
    except Exception:
        pass """
    """ print("Puntos antes de convertir", last_pts)
    print("Marcadores detectados", all_markers_detected) """

    for i in range(len(marker_map)):
        #if i == last_changed[0] or all(all_markers_detected):
        last_pts[i+1] = last_pts[i+1]/ 100.0

    vias_local = last_pts.copy()
    if vias_local.ndim == 1:
        vias_local = vias_local.reshape(-1, 3)
    Y_obs = tf.expand_dims(tf.convert_to_tensor(vias_local, dtype=default_float()), axis=0)

    obs_idx = [int(np.argmin(np.abs(test_x - t))) for t in via_t_local[:, 0]]
    F_obs = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()), axis=0)

    try:
        dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)
        X_test = tf.convert_to_tensor(test_x.reshape(-1, 1), dtype=default_float())
        dF_full = dF_fn(X_test)
        F_cond = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0) + dF_full
        mean_corrected = F_cond.numpy()
        mean_combo = np.squeeze(mean_corrected)
    except Exception as exc:
        mean_combo = mean_gmp_arr.T.copy()

    # orientation conditioning if available
    if mean_quat_arr is not None and last_quats is not None:
        q_local = last_quats.copy()
        if q_local.ndim == 1:
            q_local = q_local.reshape(-1,4)
        Y_obs_q = tf.expand_dims(tf.convert_to_tensor(q_local, dtype=default_float()), axis=0)
        F_obs_q = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T[obs_idx], dtype=default_float()), axis=0)
        try:
            dF_fn_q = updates.exact(kernel_q, X_obs, Y_obs_q, F_obs_q, diag=0.0)
            X_test = tf.convert_to_tensor(test_x.reshape(-1, 1), dtype=default_float())
            dF_full_q = dF_fn_q(X_test)
            F_prior_q = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T, dtype=default_float()), axis=0)
            F_cond_q = F_prior_q + dF_full_q
            mean_cond_q = np.squeeze(F_cond_q.numpy())
        except Exception:
            mean_cond_q = mean_quat_arr.T.copy()
    else:
        mean_cond_q = None

    # plotting (mm)
    try:
        global last_final_line, prior_line
        if last_final_line is not None:
            try: last_final_line.remove()
            except Exception: pass
            last_final_line = None
        if prior_line is not None:
            try: prior_line.remove()
            except Exception: pass
            prior_line = None

        prior_line, = ax.plot(mean_gmp_arr[0], mean_gmp_arr[1], mean_gmp_arr[2], '--', color='blue', linewidth=2, label='GMP Prior')
        last_final_line, = ax.plot(mean_combo[:, 0], mean_combo[:, 1], mean_combo[:, 2], '-.', color='red', linewidth=3, label='Final')

        # draw orientation frames if available
        if mean_cond_q is not None:
            colors_vec = ['r','g','b']
            for i_pt, t_idx in enumerate(obs_idx):
                if t_idx < 0 or t_idx >= mean_combo.shape[0]:
                    continue
                qv = mean_cond_q[t_idx]
                try:
                    Rm = R.from_quat(qv).as_matrix()
                except Exception:
                    continue
                pt = mean_combo[t_idx]
                #print("Valores de pt", pt)
                for j, col in enumerate(colors_vec):
                    try:
                        art = ax.quiver(pt[0], pt[1], pt[2],
                                        Rm[0, j], Rm[1, j], Rm[2, j],
                                        length=arrow_len, color=col, normalize=True)
                        frame_artists.append(art)
                    except Exception:
                        pass

        # adjust limits
        try:
            all_x = np.hstack([mean_gmp_arr[0], mean_combo[:, 0], vias[:, 0]])
            all_y = np.hstack([mean_gmp_arr[1], mean_combo[:, 1], vias[:, 1]])
            all_z = np.hstack([mean_gmp_arr[2], mean_combo[:, 2], vias[:, 2]])
            margin = 0.05
            def minmax_with_margin(arr):
                mn = np.min(arr); mx = np.max(arr)
                rng = max(1e-6, mx - mn)
                return mn - margin * rng, mx + margin * rng
            ax.set_xlim(minmax_with_margin(all_x))
            ax.set_ylim(minmax_with_margin(all_y))
            ax.set_zlim(minmax_with_margin(all_z))
        except Exception:
            pass

        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]')
        try: ax.legend()
        except Exception: pass
        plt.draw()
    except Exception:
        pass

    # update pb path (mm->cm) y path de orientaciones
    if pb_global is not None:
        try:
            # posiciones
            pb_global.set_path(mean_combo * 10.0)
            # orientaciones: mean_cond_q tiene shape (Nt,4) si existe y está alineado con mean_combo
            if mean_cond_q is not None:
                try:
                    # asegurar shape (Nt,4)
                    mq = np.asarray(mean_cond_q, dtype=float)
                    if mq.ndim == 2 and mq.shape[1] == 4:
                        pb_global.set_orientations(mq)
                except Exception:
                    pass
        except Exception:
            pass

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    dt = 0.01
    gap = 15
    demostraciones = 1

    use_csv_quats = False
    Q = None

    # --- Load CSVs with pos + quat if present ---
    try:
        csv_files = [f for f in os.listdir(CSV_FOLDER) if f.startswith("obstacle_") and f.endswith(".csv")]
        if len(csv_files) >= 1:
            values = []
            lengths = []
            for i in range(1, demostraciones + 1):
                fn = os.path.join(CSV_FOLDER, "pick1.csv")
                data = pd.read_csv(fn)
                lengths.append(len(data['x'][::gap]))
            min_length = min(lengths)
            Q_list = []
            for i in range(1, demostraciones + 1):
                fn = os.path.join(CSV_FOLDER, "pick1.csv")
                data = pd.read_csv(fn)
                x_data = np.array(data['x'][::gap])[:min_length]*10
                y_data = np.array(data['y'][::gap])[:min_length]*10
                z_data = np.array(data['z'][::gap])[:min_length]*10
                # quaternion columns (if present)
                if all(c in data.columns for c in ['qx','qy','qz','qw']):
                    qx_data = np.array(data['qx'][::gap])[:min_length]
                    qy_data = np.array(data['qy'][::gap])[:min_length]
                    qz_data = np.array(data['qz'][::gap])[:min_length]
                    qw_data = np.array(data['qw'][::gap])[:min_length]
                else:
                    qx_data = np.zeros(min_length)
                    qy_data = np.zeros(min_length)
                    qz_data = np.zeros(min_length)
                    qw_data = np.ones(min_length)

                values.append(np.array([x_data, y_data, z_data]))
                quat_vals = np.array([qx_data, qy_data, qz_data, qw_data]).T  # (T,4)

                if i == 1:
                    Q = quat_vals.copy()
                    X = np.linspace(0, (min_length-1)*dt, min_length).reshape(-1,1)
                    Y = np.vstack((np.array([x_data, y_data, z_data]).T))
                else:
                    Q = np.vstack((Q, quat_vals))
                    X = np.vstack((X, np.linspace(0, (min_length-1)*dt, min_length).reshape(-1,1)))
                    Y = np.vstack((Y, np.array([x_data, y_data, z_data]).T))
            use_csv_quats = True
        else:
            use_csv_quats = False
    except Exception as e:
        print("CSV loading error:", e)
        use_csv_quats = False

    # fallback if no CSVs: create trivial task
    if not use_csv_quats:
        test_x = np.arange(0.0, 1.0, dt)
        via_t = np.array([0.5, 1.0]).reshape(-1,1)
        vias = np.array([[0,0,0],[50,50,50],[100,0,0]])
        vias_quat = None
    else:
        min_length = Q.shape[0] // demostraciones
        target_t = (min_length-1) * dt
        test_x = np.arange(0.0, target_t + dt, dt)
        pos0 = values[0]
        target_position = np.array([pos0[0][-1], pos0[1][-1], pos0[2][-1]])
        middle_position = np.array([pos0[0][min_length//2], pos0[1][min_length//2], pos0[2][min_length//2]])
        via_point0_position = np.array([pos0[0][0], pos0[1][0], pos0[2][0]])
        via_t = np.array([0.0, (min_length//2)*dt, target_t]).reshape(-1,1)
        vias = np.array([via_point0_position, middle_position, target_position])
        min_idx = min_length // 2
        vias_quat = np.array([
            [Q[0,0], Q[0,1], Q[0,2], Q[0,3]],
            [Q[min_idx,0], Q[min_idx,1], Q[min_idx,2], Q[min_idx,3]],
            [Q[min_length-1,0], Q[min_length-1,1], Q[min_length-1,2], Q[min_length-1,3]],
        ])

    # --- GP for position ---
    size = Y.shape[0] if 'Y' in locals() else vias.shape[0]
    observation_noise = 0.5
    gp_mp = ProGpMp(X, Y, via_t, vias, dim=3, demos=demostraciones, size=size, observation_noise=observation_noise)
    gp_mp.BlendedGpMp(gp_mp.ProGP)
    mean_gmp, var_blended = gp_mp.predict_BlendedPos(test_x.reshape(-1,1))
    mean_gmp_arr = np.vstack(mean_gmp)   # (3, Nt) in mm

    # --- GP for quaternion if CSV provided ---
    mean_quat_arr = None
    if use_csv_quats and Q is not None:
        try:
            gp_quat = ProGpMp(X, Q, via_t, vias_quat, dim=4, demos=demostraciones, size=Q.shape[0]//demostraciones, observation_noise=0.01)
            gp_quat.BlendedGpMp(gp_quat.ProGP)
            mean_quat, var_quat = gp_quat.predict_BlendedPos(test_x.reshape(-1,1))
            mean_quat_arr = np.vstack(mean_quat)   # (4, Nt)
        except Exception as e:
            print("gp_quat error:", e)
            mean_quat_arr = None

    # --- initialize GUI and PyBullet ---
    last_pts = vias.copy()   # mm
    if mean_quat_arr is not None:
        last_quats = vias_quat.copy()
    else:
        last_quats = np.tile(np.array([0.0,0.0,0.0,1.0]), (vias.shape[0],1))

    dp3 = CameraOnlyPoints3D(last_pts, on_update=print_update_3d)
    ax = dp3.ax

    last_changed = [0]
    try:
        pathwise_update_fn()
    except Exception:
        pass

    plt.show(block=False)

    # start PyBulletManager (try to start ADAM if available)
    try:
        pb_global = PyBulletManager(start_adam=True, adam_urdf=ADAM_URDF_PATH, use_gui=True)
        initial_positions_cm = (vias * 10.0).copy()
        pb_global.create_spheres(initial_positions_cm, radius_cm=5.0, rgba=[0.8,0.2,0.2,0.5])
        # si mean_gmp_arr existe la ponemos como path inicial
        pb_global.set_path(mean_gmp_arr.T * 10.0)
        # si mean_quat_arr existe la guardamos también (se normaliza en set_orientations)
        if mean_quat_arr is not None:
            try:
                mq = mean_quat_arr.T.copy()  # mean_quat_arr shape (4,Nt) -> transpose (Nt,4)
                pb_global.set_orientations(mq)
            except Exception:
                pass

        # start ADAM follower if ADAM available
        if getattr(pb_global, 'adam', None) is not None:
            try:
                pb_global.start_adam_follow(speed=20.0, ee_side='left', ee_link_name='dummy')
            except Exception:
                pass
    except Exception as exc:
        print("Error inicializando PyBulletManager:", exc)
        pb_global = None

    # Camera / ArUco params (ajusta a tu cámara)
    camera_matrix = np.array([
        [910.1155107777962, 0.0, 360.3277519024787],
        [0.0, 910.2233367566544, 372.6634999577232],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([0.0212284835698144, 0.8546829039917951, 0.0034281408326615323, 0.0005749116561059772, -3.217248182814475])

    # mapping markers -> indices (ejemplo)
    marker_map = {66:1, 24: 2}
    marker_lengths = {66: 0.053, 24: 0.063, "default": 0.053}
    all_markers_detected = [False] * len(marker_map)  # track which markers have been seen at least once

    try:
        dp3.start_camera_tracking(
            vs_src=6,   # ajusta a tu cámara (0 para built-in, o índice/vídeo)
            aruco_type=cv2.aruco.DICT_5X5_100,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            marker_length_m=marker_lengths,
            tracked_idx=marker_map,
            apply_fixed_rotations=True,
            video_w=1000
        )
    except Exception as e:
        print("start_camera_tracking error (continuamos sin cámara):", e)

    try:
        dp3.show()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            dp3.stop_camera()
        except Exception:
            pass
        try:
            if pb_global is not None:
                pb_global.stop(disconnect_client=True)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            plt.close('all')
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass
