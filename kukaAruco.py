#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from scipy.io import loadmat
import pandas as pd
import time
# GPflowSampling imports
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import priors, updates, decoupled
from gpflow_sampling.sampling.core import AbstractSampler

from ProGP import ProGpMp

# --- Added imports for PyBullet, threading and camera ---
import threading
import pybullet as p
import pybullet_data

# ArUco / camera
from imutils.video import VideoStream
import imutils
import cv2
from scipy.spatial.transform import Rotation as R

# --------------------------
# Variables globales
# --------------------------
last_changed = None
last_pts = None
marker_map =None  # mapping markerID -> via-point index

mean_gmp_arr = np.array([])
vias = np.array([])
via_t = np.array([])
last_final_line, prior_line = None, None
ax = None

pb = None  # handler PyBullet (se inicializa en main)
test_x = None  # grid temporal (se define en main)

# Globals for camera->pybullet orientation update
camera_last_quat = None   # quaternion (x,y,z,w) from last detection
camera_last_pos_mm = None # last detected position in mm
camera_tracked_idx = 0    # which sphere index the camera controls

# --------------------------
# PyBullet handler
# --------------------------
class PyBulletSpheres:
    def __init__(self, use_gui=True, gravity=-9.81, time_step=1./240.):
        if p is None:
            raise ImportError("pybullet no está instalado. Instálalo con: pip install pybullet")
        self.client = None
        self._sim_thread = None
        self._sim_running = False
        self.bodies = []
        self.lock = threading.Lock()
        self.time_step = time_step

        flags = p.GUI if use_gui else p.DIRECT
        self.client = p.connect(flags)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, gravity, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)
        _ = p.loadURDF("plane.urdf", physicsClientId=self.client)

        # debug lines ids for drawing the path
        self.path_ids = []

        # iniciar el loop de simulación
        self._sim_running = True
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

        # robot & follow-thread
        self.robot_id = None
        self.robot_joints = []
        self.ee_link = 6
        self.path_m = None        # path en metros (Nx3) o None
        self.path_version = 0
        self._robot_thread = None
        self._robot_running = False

    def _sim_loop(self):
        while self._sim_running:
            with self.lock:
                p.stepSimulation(physicsClientId=self.client)
            time.sleep(self.time_step)

    # --- Esferas ---
    def create_spheres(self, positions_cm, radius_cm=1.0, mass=0.0, rgba=[1,0,0,1]):
        with self.lock:
            for b in self.bodies:
                try:
                    p.removeBody(b, physicsClientId=self.client)
                except Exception:
                    pass
            self.bodies = []

            positions_m = np.array(positions_cm, dtype=float) / 100.0
            radius_m = float(radius_cm) / 100.0

            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius_m, rgbaColor=rgba, physicsClientId=self.client)

            for pos in positions_m:
                body = p.createMultiBody(baseMass=mass,
                                        baseVisualShapeIndex=vis,
                                        basePosition=[float(pos[0]), float(pos[1]), float(pos[2])],
                                        physicsClientId=self.client)
                self.bodies.append(body)
        return self.bodies

    def update_spheres(self, positions_cm):

        with self.lock:
            if not self.bodies:
                return

            # Caso dict: actualizar sólo los índices especificados
            if isinstance(positions_cm, dict):
                for idx, pos in positions_cm.items():
                    try:
                        i = int(idx)
                        if 0 <= i < len(self.bodies):
                            p.resetBasePositionAndOrientation(
                                self.bodies[i],
                                [float(pos[0]) / 100.0, float(pos[1]) / 100.0, float(pos[2]) / 100.0],  # cm->m
                                [0, 0, 0, 1],
                                physicsClientId=self.client
                            )
                    except Exception:
                        continue
                return

            # Array/list: normalizar a Nx3
            positions_arr = np.array(positions_cm, dtype=float)
            if positions_arr.ndim == 1 and positions_arr.size == 3:
                positions_arr = positions_arr.reshape(1, 3)

            if positions_arr.ndim == 2 and positions_arr.shape[1] == 3:
                n = positions_arr.shape[0]
                # Si N coincide con bodies -> actualizar todas
                if n == len(self.bodies):
                    for bid, pos in zip(self.bodies, positions_arr):
                        try:
                            p.resetBasePositionAndOrientation(bid,
                                [float(pos[0]) / 100.0, float(pos[1]) / 100.0, float(pos[2]) / 100.0],
                                [0,0,0,1], physicsClientId=self.client)
                        except Exception:
                            pass
                    return
                # Si difiere, actualizamos solo el prefijo mínimo (no recrear)
                m = min(n, len(self.bodies))
                for bid, pos in zip(self.bodies[:m], positions_arr[:m]):
                    try:
                        p.resetBasePositionAndOrientation(bid,
                            [float(pos[0]) / 100.0, float(pos[1]) / 100.0, float(pos[2]) / 100.0],
                            [0,0,0,1], physicsClientId=self.client)
                    except Exception:
                        pass
                return

            # formato no reconocido -> ignorar
            return

    # --- Robot ---
    def load_kuka(self, base_pos=[0,0,0], urdf_rel_path="kuka_iiwa/model.urdf"):
        with self.lock:
            kukaId = p.loadURDF(urdf_rel_path, base_pos, useFixedBase=True, physicsClientId=self.client)
            self.robot_id = kukaId
            numJoints = p.getNumJoints(kukaId, physicsClientId=self.client)
            self.robot_joints = [j for j in range(numJoints)
                                if p.getJointInfo(kukaId, j, physicsClientId=self.client)[2] == p.JOINT_REVOLUTE]
            rp = [0, 0, 0, 0.5 * math.pi, 0, -0.5 * math.pi * 0.66, 0]
            for i, j in enumerate(self.robot_joints):
                try:
                    angle = rp[i] if i < len(rp) else 0.0
                    p.resetJointState(self.robot_id, j, angle, physicsClientId=self.client)
                except Exception:
                    pass

    def set_path(self, path_cm):
        with self.lock:
            if path_cm is None:
                self.path_m = None
            else:
                arr = np.array(path_cm, dtype=float)
                self.path_m = arr * 0.01  # cm -> m
            self.path_version += 1

    def draw_path(self, points_cm, step=5, color=[1,0,0], width=2.0, lifeTime=0):
        with self.lock:
            for pid in self.path_ids:
                try:
                    p.removeUserDebugItem(pid, physicsClientId=self.client)
                except Exception:
                    pass
            self.path_ids = []

            if points_cm is None:
                return
            pts = np.array(points_cm, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3:
                return
            n = pts.shape[0]
            if n < 2:
                return
            pts_m = pts * 0.01
            for i in range(0, n-1, step):
                a = pts_m[i].tolist()
                b = pts_m[min(i+step, n-1)].tolist()
                try:
                    pid = p.addUserDebugLine(a, b, lineColorRGB=color, lineWidth=width, lifeTime=lifeTime, physicsClientId=self.client)
                    self.path_ids.append(pid)
                except Exception:
                    pass

    def start_robot_follow(self, speed=1.0, ee_link=None):
        if ee_link is not None:
            self.ee_link = ee_link
        if self.robot_id is None:
            raise RuntimeError("Carga primero el robot con load_kuka()")
        if self._robot_running:
            return
        self._robot_running = True
        self._robot_thread = threading.Thread(target=self._robot_loop, args=(speed,), daemon=True)
        self._robot_thread.start()

    def stop_robot_follow(self):
        self._robot_running = False
        if self._robot_thread is not None:
            self._robot_thread.join(timeout=1.0)
            self._robot_thread = None

    def _get_ee_position(self):
        try:
            st = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True, physicsClientId=self.client)
            if len(st) >= 5 and st[4] is not None:
                return np.array(st[4], dtype=float)
            else:
                return np.array(st[0], dtype=float)
        except Exception:
            return None

    def _robot_loop(self, speed):
        while self._robot_running:
            with self.lock:
                local_path = None if self.path_m is None else (self.path_m.copy())
                local_version = self.path_version

            if local_path is None or local_path.size == 0:
                time.sleep(0.05)
                continue

            ee_pos = self._get_ee_position()
            if ee_pos is None:
                idx0 = 0
            else:
                dists = np.linalg.norm(local_path - ee_pos.reshape(1,3), axis=1)
                idx0 = int(np.argmin(dists))
            i = idx0
            while i < len(local_path) and self._robot_running:
                with self.lock:
                    if local_version != self.path_version:
                        break
                    path_copy = self.path_m.copy() if self.path_m is not None else None

                if path_copy is None or path_copy.size == 0:
                    break

                target = path_copy[i].tolist()
                with self.lock:
                    try:
                        joint_positions = p.calculateInverseKinematics(
                            self.robot_id,
                            self.ee_link,
                            targetPosition=target,
                            physicsClientId=self.client
                        )
                        p.setJointMotorControlArray(
                            self.robot_id,
                            jointIndices=self.robot_joints,
                            controlMode=p.POSITION_CONTROL,
                            targetPositions=joint_positions[:len(self.robot_joints)],
                            physicsClientId=self.client
                        )
                    except Exception:
                        pass

                time.sleep(self.time_step * max(0.01, speed))
                i += 1
            time.sleep(0.005)

    def stop(self):
        self.stop_robot_follow()
        self._sim_running = False
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=1.0)
        if self.client is not None:
            try:
                p.disconnect(self.client)
            except Exception:
                pass
            self.client = None

# --------------------------
# Callback integrado (solo cámara -> pathwise)
# --------------------------
def print_update_3d(pts, changed):
    global last_changed, last_pts, pb
    last_changed = changed
    last_pts = pts.copy()

    # Actualizar solo índices changed (si quieres hacerlo aquí en bloque)
    if pb is not None and changed is not None:
        try:
            upd = {}
            for i in changed:
                upd[int(i)] = (pts[int(i)] / 10.0).tolist()
            if upd:
                pb.update_spheres(upd)
        except Exception:
            pass

    if last_changed is not None:
        pathwise_update_fn()

def pathwise_update_fn():
    global mean_gmp_arr, vias, via_t, last_changed, last_pts, last_final_line, prior_line, ax, pb, test_x, marker_map

    BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
    base_kernels = [cls(lengthscales=1.0) for cls in BaseClasses]
    kernel = kernels.SeparateIndependent(base_kernels)

    # via_t_local shape (3,1)
    via_t_local = np.array([
        via_t[0],
        via_t[-1] / 2,
        via_t[-1]
    ])

    X_obs = tf.convert_to_tensor(via_t_local, dtype=default_float())
    
    for i in range(len(marker_map)):
        last_pts[i] = last_pts[i]/ 100.0
    #last_pts[1] = last_pts[1]/ 100.0
    
    #print("Last points changed (mm):", last_pts)

    vias_local = last_pts.copy()  # last_pts en mm
    if vias_local.ndim == 1:
        vias_local = vias_local.reshape(-1, 3)
    Y_obs = tf.expand_dims(tf.convert_to_tensor(vias_local, dtype=default_float()), axis=0)

    obs_idx = [int(np.argmin(np.abs(test_x - t))) for t in via_t_local[:, 0]]
    F_obs = tf.expand_dims(
        tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()),
        axis=0
    )

    dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)

    X_test = tf.convert_to_tensor(test_x.reshape(-1, 1), dtype=default_float())
    dF_full = dF_fn(X_test)
    F_cond = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0) + dF_full

    mean_corrected = F_cond.numpy()
    mean_combo = np.squeeze(mean_corrected)

    try:
        global last_final_line, prior_line
        if last_final_line is not None:
            try:
                last_final_line.remove()
            except Exception:
                pass
        if prior_line is not None:
            try:
                prior_line.remove()
            except Exception:
                pass

        # Plot prior & conditioned path (both in mm) on global ax
        prior_line, = ax.plot(mean_gmp_arr[0], mean_gmp_arr[1], mean_gmp_arr[2], '--', color='blue', linewidth=2, label='GMP Prior')
        last_final_line, = ax.plot(mean_combo[:, 0], mean_combo[:, 1], mean_combo[:, 2], '-.', color='red', linewidth=3, label='Final')

        # Ajustar límites para que se vean ambas líneas y los via points
        try:
            all_x = np.hstack([mean_gmp_arr[0], mean_combo[:, 0], vias[:, 0]])
            all_y = np.hstack([mean_gmp_arr[1], mean_combo[:, 1], vias[:, 1]])
            all_z = np.hstack([mean_gmp_arr[2], mean_combo[:, 2], vias[:, 2]])
            margin = 0.05  # 5%
            def minmax_with_margin(arr):
                mn = np.min(arr); mx = np.max(arr)
                rng = max(1e-6, mx - mn)
                return mn - margin * rng, mx + margin * rng
            ax.set_xlim(minmax_with_margin(all_x))
            ax.set_ylim(minmax_with_margin(all_y))
            ax.set_zlim(minmax_with_margin(all_z))
        except Exception:
            pass

        ax.set_xlabel('x [mm]', fontsize=12)
        ax.set_ylabel('y [mm]', fontsize=12)
        ax.set_zlabel('z [mm]', fontsize=12)
        ax.legend(fontsize=12)
        plt.draw()
    except Exception:
        pass

    # actualizar path en PyBullet (no bloqueante)
    if pb is not None:
        try:
            pb.set_path(mean_combo * 10.0)   # mean_combo en mm -> *10 => cm
        except Exception:
            pass

# --------------------------
# Camera-only GUI: no mouse drag, solo visualización + updates desde cámara
# --------------------------
class CameraOnlyPoints3D:
    """
    Visualización 3D (sin drag). Trabaja en MM. on_update callback recibe (points_mm, changed_idxs).
    """
    def __init__(self, points_mm, on_update=None, fig_size=(8, 8)):
        self.points = np.array(points_mm, float)  # Nx3 en mm
        self.last_points = self.points.copy()
        self.on_update = on_update

        self.fig = plt.figure(figsize=fig_size)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.mouse_init(rotate_btn=3, zoom_btn=2)

        xs, ys, zs = self.points.T
        self.scatter = self.ax.scatter(xs, ys, zs, s=100, c='red')

        self.texts = []
        for i, (x, y, z) in enumerate(self.points):
            txt = self.ax.text(x, y, z, str(i + 1),
                            color='white', ha='center', va='center',
                            bbox=dict(boxstyle='circle', fc='blue', alpha=0.6))
            self.texts.append(txt)

        self.ax.set_title('Camera-controlled via-points (mm) — no mouse drag')

        # camera thread
        self._cam_thread = None
        self._cam_running = False
        self._last_pose_mm = None
        self._last_quat = None

    # notify same semantic as original (points in MM)
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
        global camera_last_quat, camera_last_pos_mm, camera_tracked_idx, pb
        try:
            # mantener datos en mm
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
            camera_last_pos_mm = new_point_mm.copy()
            camera_tracked_idx = idx

            # ACTUALIZACIÓN POR ÍNDICE EN PYPB: pasar pos en cm (release: update_spheres soporta dict)
            if pb is not None:
                try:
                    # pb expects cm in update_spheres; here pasamos dict {idx: [x_cm,y_cm,z_cm]}
                    pb.update_spheres({int(idx): (new_point_mm / 10.0).tolist()})
                    # si también queremos aplicar orientación:
                    with pb.lock:
                        if 0 <= idx < len(pb.bodies):
                            pos_m = (new_point_mm / 1000.0).tolist()  # mm->m
                            q = quat.tolist() if quat is not None else [0,0,0,1]
                            p.resetBasePositionAndOrientation(pb.bodies[idx], pos_m, q, physicsClientId=pb.client)
                except Exception:
                    pass

            # notify -> triggers print_update_3d -> pathwise
            self._notify()
            self.fig.canvas.draw_idle()
        except Exception as e:
            print("update_point_mm error:", e)

    def start_camera_tracking(self,
                            vs_src=0,
                            aruco_type=cv2.aruco.DICT_5X5_100,
                            camera_matrix=None,
                            dist_coeffs=None,
                            marker_length_m=0.053,
                            tracked_idx=0,
                            apply_fixed_rotations=True,
                            video_w=1000):
        """
        Lanza hilo que detecta ArUco y actualiza self.points[tracked_idx] en MM.
        Robust: soporta distintas versiones de OpenCV y fallback a cv2.VideoCapture.
        """
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
            # preparar detector ArUco (clásico)
            arucoDict = cv2.aruco.getPredefinedDictionary(self._cam_aruco_type)
            if hasattr(cv2.aruco, 'DetectorParameters_create'):
                arucoParams = cv2.aruco.DetectorParameters_create()
            else:
                arucoParams = cv2.aruco.DetectorParameters()

            # inicializar captura (VideoStream con fallback a VideoCapture)
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

            # Si self._cam_marker_len es dict -> mapping markerID -> length_m
            default_marker_length_m = None
            if isinstance(self._cam_marker_len, dict):
                # si hay una entrada "default" úsala
                default_marker_length_m = self._cam_marker_len.get("default", None)
            else:
                # si es float/int, mantenerlo como valor por defecto
                try:
                    default_marker_length_m = float(self._cam_marker_len)
                except Exception:
                    default_marker_length_m = None

            while self._cam_running:
                # leer frame
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

                # detect markers
                try:
                    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
                except Exception:
                    corners, ids, rejected = [], None, None

                if ids is not None and len(corners) > 0:
                    # normalizar ids a 1D
                    try:
                        ids_flat = ids.flatten()
                    except Exception:
                        ids_flat = np.array(ids).reshape(-1)

                    # recorrer todos los marcadores detectados
                    for (markerCorner, markerID) in zip(corners, ids_flat):
                        try:
                            # corners vienen en orden TL,TR,BR,BL
                            corners_pts = markerCorner.reshape((4, 2))
                            (topLeft, topRight, bottomRight, bottomLeft) = corners_pts

                            # decidir marker_length para ESTE marker
                            if isinstance(self._cam_marker_len, dict):
                                # int(markerID) si las claves son enteros
                                marker_len_m = self._cam_marker_len.get(int(markerID), default_marker_length_m)
                            else:
                                # si self._cam_marker_len es un scalar (float) o lista
                                if isinstance(self._cam_marker_len, (list, tuple, np.ndarray)):
                                    # si es lista y su longitud coincide con número de via-points
                                    # intentamos mapear por posición del marker en la detección
                                    try:
                                        # buscar posición en ids_flat
                                        pos_in_detect = list(ids_flat).index(int(markerID))
                                        if pos_in_detect < len(self._cam_marker_len):
                                            marker_len_m = float(self._cam_marker_len[pos_in_detect])
                                        else:
                                            marker_len_m = default_marker_length_m
                                    except Exception:
                                        marker_len_m = default_marker_length_m
                                else:
                                    marker_len_m = default_marker_length_m

                            if marker_len_m is None:
                                # no hay longitud conocida: saltar este marcador
                                # (podrías usar un fallback constante aquí si prefieres)
                                continue

                            # calcular pose del marcador CON SU longitud específica
                            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                                [markerCorner], marker_len_m, self._cam_camMat, self._cam_dist
                            )
                            tvec = tvecs[0][0]   # metros
                            rvec = rvecs[0][0]

                            # Convert rvec -> rot matrix
                            R_camera, _ = cv2.Rodrigues(rvec)

                            # Build homogeneous transform T_marker_in_camera
                            T_marker_in_camera = np.eye(4)
                            T_marker_in_camera[:3, :3] = R_camera
                            T_marker_in_camera[:3, 3] = tvec

                            # Aplicar transformaciones fijas si corresponde
                            if self._cam_apply_fixrot:
                                T_base_shoulder = make_transform([0.007, 0, 0.35], [0, 0, 0])
                                T_shoulder_joint = make_transform([0.115, 0, 0.030], [0, 0, 0])
                                T_joint_support = make_transform([0.027, 0, 0.0425], [0, 0, 0])
                                T_support_camera = make_transform([0.01803, 0.0175, 0], [-np.pi/2, 0, -np.pi/2])
                                T_marker_in_robot = T_base_shoulder @ T_shoulder_joint @ T_joint_support @ T_support_camera @ T_marker_in_camera
                            else:
                                R_y = R.from_euler('y', -90, degrees=True).as_matrix()
                                R_z = R.from_euler('z', 180, degrees=True).as_matrix()
                                T_marker_in_robot = T_marker_in_camera.copy()
                                T_marker_in_robot[:3, :3] = R_z @ R_y @ T_marker_in_camera[:3, :3]

                            # extraer posición y orientación (pos en metros)
                            pos_m = T_marker_in_robot[:3, 3]
                            pos_mm = pos_m * 1000.0  # mm
                            rot_mat = T_marker_in_robot[:3, :3]
                            rot = R.from_matrix(rot_mat)
                            quat = rot.as_quat()  # (x,y,z,w)

                            # decide qué via-point actualizar (mapeo ya en tu código)
                            dest_idx = None
                            idx_map = self._cam_tracked_idx
                            # - soporta dict {markerID: via_index}
                            if isinstance(idx_map, dict):
                                dest_idx = idx_map.get(int(markerID), None)
                            else:
                                # otros comportamientos: int base, list, ...
                                try:
                                    base = int(idx_map)
                                    # fallback: base + position_in_detection
                                    try:
                                        k = list(ids_flat).index(int(markerID))
                                    except Exception:
                                        k = 0
                                    dest_idx = base + k
                                except Exception:
                                    # si markerID es en sí un índice válido
                                    try:
                                        mid = int(markerID)
                                        if 0 <= mid < self.points.shape[0]:
                                            dest_idx = mid
                                    except Exception:
                                        dest_idx = None

                            if dest_idx is None or not (0 <= dest_idx < self.points.shape[0]):
                                # si no mapeado -> saltar
                                continue

                            # actualizar el via-point (mm)
                            self.update_point_mm(dest_idx, pos_mm, quat=quat, set_text=True)

                            # dibujo de debug
                            try:
                                cv2.drawFrameAxes(frame, self._cam_camMat, self._cam_dist, rvec, tvec, marker_len_m)
                            except Exception:
                                pass

                            # dibujar caja y texto
                            try:
                                topRight = (int(topRight[0]), int(topRight[1]))
                                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                                topLeft = (int(topLeft[0]), int(topLeft[1]))
                            except Exception:
                                topLeft = tuple(corners_pts[0].astype(int))
                                topRight = tuple(corners_pts[1].astype(int))
                                bottomRight = tuple(corners_pts[2].astype(int))
                                bottomLeft = tuple(corners_pts[3].astype(int))

                            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

                            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

                            cv2.putText(frame, f"ID:{int(markerID)}->pt:{dest_idx}",
                                        (topLeft[0], topLeft[1] - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        except Exception:
                            # si hay fallo con este marcador, continuar con el resto
                            continue

                # mostrar frame
                try:
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self._cam_running = False
                        break
                except Exception:
                    pass

                # give CPU a little break
                time.sleep(0.001)

            # cleanup capture objects
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

        # start thread
        self._cam_thread = threading.Thread(target=_cam_loop, daemon=True)
        self._cam_thread.start()

    def stop_camera(self):
        self._cam_running = False
        if self._cam_thread is not None:
            self._cam_thread.join(timeout=1.0)
            self._cam_thread = None

# --------------------------
# Helper para build transform (usada en tu script aruco)
# --------------------------
def make_transform(t, rpy):
    rot = R.from_euler('xyz', rpy)
    R_mat = rot.as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

def import_object(path, obj_pose):
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


if __name__ == '__main__':
    # --- Cargo los datos RAIL tal y como tenías en tu script ---
    def leer_archivos_mat(ruta_carpeta, number):
        data = {}
        for archivo in os.listdir(ruta_carpeta):
            if archivo.endswith(str(number)+'.mat'):
                nombre_archivo, extension = os.path.splitext(archivo)
                try:
                    clave = int(nombre_archivo)
                except ValueError:
                    continue
                ruta_archivo = os.path.join(ruta_carpeta, archivo)
                datos = loadmat(ruta_archivo)
                data[clave] = datos
        return data

    ruta_carpeta = '/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/RAIL/REACHING'
    number = 6
    data = leer_archivos_mat(ruta_carpeta, number)

    dt = 0.01
    gap = 30

    demostraciones = 1
    demos = data[number]['dataset']
    for i in range(demostraciones):
        demo = demos[i]
        pos = demo['pos'][0].T[:, 0::gap]*10
        vel = demo['vel'][0].T[:, 0::gap]
        acc = demo['acc'][0].T[:, 0::gap]
        t = demo['time'][0].T[:, 0::gap]
        via_t = t.T
        vias = pos.T
        if i == 0:
            size = vias.shape[0]
            X = via_t
            Y = vias
        else:
            X = np.vstack((X, via_t))
            Y = np.vstack((Y, vias))

    np.random.seed(30)

    target_t = sum(demos[i]['time'][0].T[:, 0::gap][0, -1] for i in range(demostraciones)) / demostraciones
    target_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, -1]*10 for i in range(demostraciones)) / demostraciones
    via_point0_t = sum(demos[i]['time'][0].T[:, 0::gap][0, 0] for i in range(demostraciones)) / demostraciones
    via_point0_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, 0]*10 for i in range(demostraciones)) / demostraciones
    via_point1_t = sum(demos[i]['time'][0].T[:, 0::gap][0, demos[i]['pos'][0].T[:, 0::gap].shape[1] * 2 // 10] for i in range(demostraciones)) / demostraciones
    via_point1_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, demos[i]['pos'][0].T[:, 0::gap].shape[1] * 2 // 10]*10 for i in range(demostraciones)) / demostraciones

    via_t = np.array([via_point0_t, target_t]).reshape(-1, 1)
    vias = np.array([via_point0_position, target_position])

    observation_noise = 1.0
    gp_mp = ProGpMp(X, Y, via_t, vias, dim=3, demos=demostraciones, size=size, observation_noise=observation_noise)
    gp_mp.BlendedGpMp(gp_mp.ProGP)
    test_x = np.arange(0.0, target_t, dt)

    alpha_list = np.ones(len(test_x))
    alpha_list = np.vstack((alpha_list, alpha_list))
    alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))

    mean_gmp, var_blended = gp_mp.predict_BlendedPos(test_x.reshape(-1, 1))

    var_blended[0] = np.where(var_blended[0] < 0, 0, var_blended[0])
    var_blended[1] = np.where(var_blended[1] < 0, 0, var_blended[1])
    var_blended[2] = np.where(var_blended[2] < 0, 0, var_blended[2])

    # ----------------------------
    # IMPORTANT: crear las 3 via-points (mm) ANTES de la GUI / PyBullet
    # ----------------------------
    vias = np.array([vias[0], vias[1] / 2.0, vias[1]])   # mm, shape (3,3)
    mean_gmp_arr = np.vstack(mean_gmp)   # shape (3, N) en mm

    # Creamos la GUI controlada por cámara (trabaja en mm) — NO permite drag con ratón
    dp3 = CameraOnlyPoints3D(vias, on_update=print_update_3d)
    # definimos ax global para pathwise_update_fn
    ax = dp3.ax

    # Forzamos un cálculo inicial del pathwise para visualizar
    last_pts = vias.copy()
    last_changed = [0]
    try:
        pathwise_update_fn()
    except Exception:
        pass

    # Subplots 2D (opcional)
    font_size = 14
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(311)
    ax2.plot(test_x, mean_gmp[0], linewidth=2)
    ax2.set_ylabel('x [mm]')
    ax3 = fig2.add_subplot(312)
    ax3.plot(test_x, mean_gmp[1], linewidth=2)
    ax3.set_ylabel('y [mm]')
    ax4 = fig2.add_subplot(313)
    ax4.plot(test_x, mean_gmp[2], linewidth=2)
    ax4.set_ylabel('z [mm]')
    plt.show(block=False)

    # Inicializar PyBullet y esferas
    try:
        pb = PyBulletSpheres(use_gui=True)
        initial_positions_cm = (vias * 10.0).copy()  # mm -> *10 => cm
        pb.create_spheres(initial_positions_cm, radius_cm=5.0, rgba=[0.8,0.2,0.2,0.5])
        pb.load_kuka(base_pos=[0,0,0], urdf_rel_path="kuka_iiwa/model.urdf")
        pb.set_path(mean_gmp_arr.T * 10)   # mean_gmp_arr en mm -> *10 => cm
        pb.start_robot_follow(speed=20, ee_link=6)
    except Exception as exc:
        print("Error inicializando PyBullet / KUKA:", exc)
        pb = None

    # ------ Setup ArUco / cámara params ------
    camera_matrix = np.array([
        [910.1155107777962, 0.0, 360.3277519024787],
        [0.0, 910.2233367566544, 372.6634999577232],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([0.0212284835698144, 0.8546829039917951, 0.0034281408326615323, 0.0005749116561059772, -3.217248182814475])
    marker_length = 0.053  # m

    # Iniciar tracking de la cámara (ajusta src si hace falta)
    marker_map = {66: 0, 42: 1} #! Change by the user
    marker_lengths = {66: 0.053, 42: 0.035, "default": 0.053}
    # m
    dp3.start_camera_tracking(
        vs_src=6,
        aruco_type=cv2.aruco.DICT_5X5_100,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_length_m=marker_lengths,
        tracked_idx=marker_map,
        apply_fixed_rotations=True,
        video_w=1000
    )

    # Mostrar la GUI (bloqueante). La cámara corre en hilo y actualiza los puntos.
    try:
        dp3.show()
    except KeyboardInterrupt:
        pass
    finally:
        # cleanup
        dp3.stop_camera()
        if pb is not None:
            pb.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
