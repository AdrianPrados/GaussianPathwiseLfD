#!/usr/bin/env python3
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import threading
import gc

# GPflow / TF
import tensorflow as tf
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import updates
from scipy.io import loadmat
import pandas as pd

# PyBullet
import pybullet as p
import pybullet_data

# Camera / ArUco
from imutils.video import VideoStream
import imutils
import cv2
from scipy.spatial.transform import Rotation as R

# ProGP (tu clase)
from ProGP import ProGpMp

# --------------------------
# Globals
# --------------------------
last_changed = None
last_pts = None          # Nx3 en mm
last_quats = None        # Nx4 (x,y,z,w)
marker_map = None        # dict markerID -> via index

mean_gmp_arr = np.array([])   # shape (3,N) mm prior
mean_quat_arr = None          # shape (4,N) prior (si existe)
vias = np.array([])
via_t = np.array([])
last_final_line, prior_line = None, None
ax = None

pb = None
test_x = None

# Last camera detection applied (used to set orientation in PyBullet per-index)
camera_last_quat = None
camera_last_pos_mm = None
camera_tracked_idx = 0

# store the quiver/axis artists so we can remove them on each update
frame_artists = []

# arrow length (mm) for coordinate frames drawn on the 3D plot
arrow_len = 1
first_time = True

# --------------------------
# PyBullet helper
# --------------------------
class PyBulletSpheres:
    def __init__(self, use_gui=True, gravity=-9.81, time_step=1./240.):
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

        self.path_ids = []
        self._sim_running = True
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

        self.robot_id = None
        self.robot_joints = []
        self.ee_link = 6
        self.path_m = None
        self.path_version = 0
        self._robot_thread = None
        self._robot_running = False

    def _sim_loop(self):
        while self._sim_running:
            with self.lock:
                try:
                    p.stepSimulation(physicsClientId=self.client)
                except Exception:
                    pass
            time.sleep(self.time_step)

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
                try:
                    body = p.createMultiBody(baseMass=mass,
                                             baseVisualShapeIndex=vis,
                                             basePosition=[float(pos[0]), float(pos[1]), float(pos[2])],
                                             physicsClientId=self.client)
                    self.bodies.append(body)
                except Exception:
                    pass
        return self.bodies

    def update_spheres(self, positions_cm):
        """
        positions_cm puede ser:
         - array Nx3 con posiciones en cm (N == len(self.bodies)), o
         - dict {index: [x_cm,y_cm,z_cm], ...} para actualizar solo índices concretos.
        """
        with self.lock:
            if not self.bodies:
                return

            # dict -> update specific indices
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

            # array/list
            positions_arr = np.array(positions_cm, dtype=float)
            if positions_arr.ndim == 1 and positions_arr.size == 3:
                positions_arr = positions_arr.reshape(1, 3)

            if positions_arr.ndim == 2 and positions_arr.shape[1] == 3:
                n = positions_arr.shape[0]
                if n == len(self.bodies):
                    for bid, pos in zip(self.bodies, positions_arr):
                        try:
                            p.resetBasePositionAndOrientation(bid,
                                [float(pos[0]) / 100.0, float(pos[1]) / 100.0, float(pos[2]) / 100.0],
                                [0,0,0,1], physicsClientId=self.client)
                        except Exception:
                            pass
                    return
                m = min(n, len(self.bodies))
                for bid, pos in zip(self.bodies[:m], positions_arr[:m]):
                    try:
                        p.resetBasePositionAndOrientation(bid,
                            [float(pos[0]) / 100.0, float(pos[1]) / 100.0, float(pos[2]) / 100.0],
                            [0,0,0,1], physicsClientId=self.client)
                    except Exception:
                        pass
                return

            return

    def load_kuka(self, base_pos=[0,0,0], urdf_rel_path="kuka_iiwa/model.urdf"):
        self._initialized = True
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

            if points_cm is None: return
            pts = np.array(points_cm, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3: return
            n = pts.shape[0]
            if n < 2: return
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

            #ee_pos = self._get_ee_position()
            #print("Me inicializo?", self._initialized)
            if self._initialized == True:
                self._initialized = False
                i = 0
            """ if ee_pos is None:
                idx0 = 0
            else:
                dists = np.linalg.norm(local_path - ee_pos.reshape(1,3), axis=1)
                idx0 = int(np.argmin(dists)) """
            #i = idx0
            
            while i < len(local_path) and self._robot_running:
                #print("Valor de idx0:", idx0)
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
# Pathwise update (pos + quat)
# --------------------------
def print_update_3d(pts, changed):
    """
    pts: Nx3 mm
    changed: list indices changed
    """
    global last_changed, last_pts, pb, camera_last_pos_mm, camera_last_quat

    last_changed = changed
    last_pts = pts.copy()

    # actualizar solo índices changed en PyBullet (si quieres)
    if pb is not None and changed is not None:
        try:
            upd = {}
            for i in changed:
                upd[int(i)] = (pts[int(i)] / 10.0).tolist()  # mm -> cm
            if upd:
                pb.update_spheres(upd)
        except Exception:
            pass

    # si la cámara ha enviado orientación/pose del tracked_idx, aplicarla al body
    if pb is not None and camera_last_pos_mm is not None and camera_last_quat is not None:
        try:
            idx = int(camera_tracked_idx)
            if 0 <= idx < len(pb.bodies):
                pos_m = (camera_last_pos_mm / 1000.0).tolist()  # mm -> m
                q = camera_last_quat.tolist()
                with pb.lock:
                    p.resetBasePositionAndOrientation(pb.bodies[idx], pos_m, q, physicsClientId=pb.client)
        except Exception:
            pass

    if last_changed is not None:
        pathwise_update_fn()

def pathwise_update_fn():
    """
    Recalcula pathwise conditioning tanto para posición (3 dims) como orientación (4 dims)
    usando las últimas observaciones (last_pts en mm, last_quats si existen).
    """
    global mean_gmp_arr, mean_quat_arr, vias, via_t, last_changed, last_pts, last_quats, ax, pb, test_x, frame_artists, last_final_line, prior_line, first_time

    # --- limpiamos frames antiguos (evitar acumulación) ---
    
    if ax is not None:
        try:
            # remove each stored artist cleanly
            for art in frame_artists:
                try:
                    art.remove()
                except Exception:
                    try:
                        # algunos quiver devuelven PathCollection u otro tipo en collections
                        if art in ax.collections:
                            ax.collections.remove(art)
                    except Exception:
                        pass
            # vaciamos lista
            frame_artists = []
        except Exception:
            frame_artists = []

    # kernels
    BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
    base_kernels = [cls(lengthscales=1.0) for cls in BaseClasses]
    kernel = kernels.SeparateIndependent(base_kernels)

    # para quaternions (4 dims) si mean_quat_arr está presente
    if mean_quat_arr is not None:
        BaseClassesQ = (kernels.SquaredExponential,)*4
        base_kernels_q = [cls(lengthscales=1.0) for cls in BaseClassesQ]
        kernel_q = kernels.SeparateIndependent(base_kernels_q)
    else:
        kernel_q = None

    # Observations times (as in tu versión: start, middle, end)
    via_t_local = np.array([via_t[0], via_t[1], via_t[-1]])
    X_obs = tf.convert_to_tensor(via_t_local, dtype=default_float())

    # COPIA local de last_pts para no mutar el global inesperadamente
    pts_copy = last_pts.copy()
    # si el antiguo código dividía por 100 en el sitio, mantenemos la conversión localmente
    try:
        if first_time == True:
            first_time = False
            for i in range(len(marker_map)):
                pts_copy[i] = pts_copy[i] / 100.0
        else:
            for i in range(min(len(pts_copy), len(marker_map) if marker_map is not None else len(pts_copy))):
                pts_copy[i] = pts_copy[i] / 100.0
    except Exception:
        pass

    vias_local = pts_copy.copy()
    if vias_local.ndim == 1:
        vias_local = vias_local.reshape(-1, 3)
    Y_obs = tf.expand_dims(tf.convert_to_tensor(vias_local, dtype=default_float()), axis=0)

    # F_obs: extraer prior at those times (mean_gmp_arr has shape (3,N))
    obs_idx = [int(np.argmin(np.abs(test_x - t))) for t in via_t_local[:, 0]]
    F_obs = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()), axis=0)

    # crear dF_fn para posición
    try:
        dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)
    except Exception as e:
        print("pathwise position dF_fn error:", e)
        dF_fn = None

    # posición: construir F_cond
    X_test = tf.convert_to_tensor(test_x.reshape(-1, 1), dtype=default_float())
    if dF_fn is not None:
        dF_full = dF_fn(X_test)
        F_cond = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0) + dF_full
        mean_corrected = F_cond.numpy()
        mean_combo = np.squeeze(mean_corrected)   # shape (Nt,3)
    else:
        mean_combo = mean_gmp_arr.T.copy()

    # --- Orientation (quaternions) pathwise (si tenemos prior) ---
    if mean_quat_arr is not None and last_quats is not None:
        q_local = last_quats.copy()
        if q_local.ndim == 1:
            q_local = q_local.reshape(-1, 4)
        Y_obs_q = tf.expand_dims(tf.convert_to_tensor(q_local, dtype=default_float()), axis=0)

        F_obs_q = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T[obs_idx], dtype=default_float()), axis=0)
        try:
            dF_fn_q = updates.exact(kernel_q, X_obs, Y_obs_q, F_obs_q, diag=0.0)
            dF_full_q = dF_fn_q(X_test)
            F_prior_q = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T, dtype=default_float()), axis=0)
            F_cond_q = F_prior_q + dF_full_q
            mean_cond_q = np.squeeze(F_cond_q.numpy())   # (Nt,4)
        except Exception as e:
            print("pathwise quat dF_fn error:", e)
            mean_cond_q = mean_quat_arr.T.copy()
    else:
        mean_cond_q = None

    # plotting (mantengo plotting en mm)
    try:
        if last_final_line is not None:
            try:
                last_final_line.remove()
            except Exception:
                pass
            last_final_line = None
        if prior_line is not None:
            try:
                prior_line.remove()
            except Exception:
                pass
            prior_line = None

        prior_line, = ax.plot(mean_gmp_arr[0], mean_gmp_arr[1], mean_gmp_arr[2], '--', color='blue', linewidth=2, label='GMP Prior')
        last_final_line, = ax.plot(mean_combo[:, 0], mean_combo[:, 1], mean_combo[:, 2], '-.', color='red', linewidth=3, label='Final')

        # also draw small coordinate frames for via quaternions if present
        if mean_cond_q is not None:
            try:
                # dibujar ejes en los puntos de observación (obs_idx)
                colors_vec = ['r','g','b']
                for i_pt, t_idx in enumerate(obs_idx):
                    if t_idx < 0 or t_idx >= mean_combo.shape[0]:
                        continue
                    qv = mean_cond_q[t_idx]   # (x,y,z,w)
                    Rm = R.from_quat(qv).as_matrix()  # 3x3
                    pt = mean_combo[t_idx]  # en mm
                    # crear tres quivers (uno por eje) y almacenar el artista para eliminar después
                    for j, col in enumerate(colors_vec):
                        try:
                            art = ax.quiver(pt[0], pt[1], pt[2],
                                            Rm[0, j], Rm[1, j], Rm[2, j],
                                            length=arrow_len, color=col, normalize=True)
                            frame_artists.append(art)
                        except Exception:
                            try:
                                art = ax.quiver(pt[0], pt[1], pt[2],
                                                float(Rm[0, j])*arrow_len, float(Rm[1, j])*arrow_len, float(Rm[2, j])*arrow_len,
                                                length=1.0, color=col)
                                frame_artists.append(art)
                            except Exception:
                                pass
            except Exception:
                pass

        # Ajustar límites
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
        try:
            ax.legend()
        except Exception:
            pass
        plt.draw()
    except Exception:
        pass

    # actualizar path en PyBullet (mean en mm -> cm para pb)
    if pb is not None:
        try:
            pb.set_path(mean_combo * 10.0)   # mm -> *10 = cm
        except Exception:
            pass

# --------------------------
# Camera GUI
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
        """
        idx: via index
        new_point_mm: numpy array [x,y,z] in mm
        quat: scipy-like quaternion (x,y,z,w)
        """
        global camera_last_quat, camera_last_pos_mm, camera_tracked_idx, pb, last_quats

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
                # store in last_quats (global)
                if last_quats is not None:
                    try:
                        last_quats[idx] = quat
                    except Exception:
                        pass

            camera_last_pos_mm = new_point_mm.copy()
            camera_tracked_idx = idx

            # Actualización por índice en PyBullet: pb.update_spheres acepta dict en cm
            if pb is not None:
                try:
                    pb.update_spheres({int(idx): (new_point_mm / 10.0).tolist()})  # mm->cm
                    # además aplicamos orientación si la tenemos
                    with pb.lock:
                        if 0 <= idx < len(pb.bodies):
                            pos_m = (new_point_mm / 1000.0).tolist()  # mm -> m
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
        
        def safe_draw_frame_axes(frame, camMat, distCoeffs, rvec, tvec, marker_len_m, margin_px=5):
            # sanity checks
            if rvec is None or tvec is None:
                return
            if np.any(np.isnan(rvec)) or np.any(np.isnan(tvec)) or np.any(np.isinf(rvec)) or np.any(np.isinf(tvec)):
                return
            h, w = frame.shape[:2]
            # puntos 3D: origen y ejes X,Y,Z del tamaño marker_len_m
            axis_pts_3d = np.float32([[0,0,0],
                                    [marker_len_m,0,0],
                                    [0,marker_len_m,0],
                                    [0,0,marker_len_m]])
            try:
                imgpts, _ = cv2.projectPoints(axis_pts_3d, rvec, tvec, camMat, distCoeffs)
                imgpts = imgpts.reshape(-1, 2)
            except Exception:
                return
            # comprobar que todos los puntos estén dentro del frame (con margen)
            inside = np.all((imgpts[:,0] >= -margin_px) & (imgpts[:,0] < w + margin_px) &
                            (imgpts[:,1] >= -margin_px) & (imgpts[:,1] < h + margin_px))
            if inside:
                safe_draw_frame_axes(frame, self._cam_camMat, self._cam_dist, rvec, tvec, marker_len_m)
                """ try:
                    cv2.drawFrameAxes(frame, camMat, distCoeffs, rvec, tvec, marker_len_m)
                except Exception:
                    pass """

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
                                T_extra = make_transform([-0.3, 0, 0], [0, 0, 0])
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
# Small helpers
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

# --------------------------
# MAIN
# --------------------------
if __name__ == '__main__':
    # --- Configura aquí paths y parámetros ---
    csv_folder = "/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/ExpCartesianADAM"
    dt = 0.01
    gap = 20
    demostraciones = 1

    use_csv_quats = False
    Q = None

    # Intentamos cargar CSV (pos+quat). Si no, fallback a RAIL .mat como antes (sin quats)
    try:
        csv_files = [f for f in os.listdir(csv_folder) if f.startswith("obstacle_") and f.endswith(".csv")]
        if len(csv_files) >= 1:
            values = []
            lengths = []
            for i in range(1, demostraciones + 1):
                fn = os.path.join(csv_folder, f"obstacle_{i}.csv")
                data = pd.read_csv(fn)
                lengths.append(len(data['x'][::gap]))
            min_length = min(lengths)
            Q_list = []
            for i in range(1, demostraciones + 1):
                fn = os.path.join(csv_folder, f"obstacle_{i}.csv")
                data = pd.read_csv(fn)
                x_data = np.array(data['x'][::gap])[:min_length]*10
                y_data = np.array(data['y'][::gap])[:min_length]*10
                z_data = np.array(data['z'][::gap])[:min_length]*10
                qx_data = np.array(data['qx'][::gap])[:min_length]
                qy_data = np.array(data['qy'][::gap])[:min_length]
                qz_data = np.array(data['qz'][::gap])[:min_length]
                qw_data = np.array(data['qw'][::gap])[:min_length]

                data_dict_pos = np.array([x_data, y_data, z_data])
                values.append(np.array([x_data, y_data, z_data]))
                quat_vals = np.array([qx_data, qy_data, qz_data, qw_data]).T  # (T,4)

                if i == 1:
                    Q = quat_vals.copy()
                    X = np.linspace(0, (min_length-1)*dt, min_length).reshape(-1,1)  # via times for training
                    Y = np.vstack((data_dict_pos.T))
                else:
                    Q = np.vstack((Q, quat_vals))
                    X = np.vstack((X, np.linspace(0, (min_length-1)*dt, min_length).reshape(-1,1)))
                    Y = np.vstack((Y, data_dict_pos.T))
            use_csv_quats = True
        else:
            use_csv_quats = False
    except Exception:
        use_csv_quats = False

    # Fallback: RAIL datasets if no CSV
    if not use_csv_quats:
        ruta_carpeta = '/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/RAIL/REACHING'
        number = 6
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
        data = leer_archivos_mat(ruta_carpeta, number)
        demos = data[number]['dataset']
        for i in range(demostraciones):
            demo = demos[i]
            pos = demo['pos'][0].T[:, 0::gap]*10
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

    # --- Preparar puntos vía (ejemplo: start, middle, target) ---
    if use_csv_quats:
        min_length = Q.shape[0] // demostraciones
        target_t = (min_length-1) * dt
        test_x = np.arange(0.0, target_t + dt, dt)
        via_point0_t = 0.0
        via_point_mid_t = (min_length//2) * dt
        via_point_target_t = target_t

        pos0 = values[0]
        target_position = np.array([pos0[0][-1], pos0[1][-1], pos0[2][-1]])
        middle_position = np.array([pos0[0][min_length//2], pos0[1][min_length//2], pos0[2][min_length//2]])
        via_point0_position = np.array([pos0[0][0], pos0[1][0], pos0[2][0]])

        via_t = np.array([via_point0_t, via_point_mid_t, via_point_target_t]).reshape(-1,1)
        vias = np.array([via_point0_position, middle_position, target_position])

        qx_data = Q[0:min_length,0] if Q is not None else None
        qy_data = Q[0:min_length,1] if Q is not None else None
        qz_data = Q[0:min_length,2] if Q is not None else None
        qw_data = Q[0:min_length,3] if Q is not None else None

        min_idx = min_length // 2
        vias_quat = np.array([
            [qx_data[0], qy_data[0], qz_data[0], qw_data[0]],
            [qx_data[min_idx], qy_data[min_idx], qz_data[min_idx], qw_data[min_idx]],
            [qx_data[-1], qy_data[-1], qz_data[-1], qw_data[-1]],
        ])
    else:
        # fallback minimal example
        test_x = np.arange(0.0, 1.0, dt)
        via_t = np.array([0.0, 0.5, 1.0]).reshape(-1,1)
        vias = np.array([[0,0,0],[50,50,50],[100,0,0]])
        vias_quat = None

    # --- Crear GP para posiciones ---
    size = Y.shape[0] if 'Y' in locals() else vias.shape[0]
    observation_noise = 0.5
    gp_mp = ProGpMp(X, Y, via_t, vias, dim=3, demos=demostraciones, size=size, observation_noise=observation_noise)
    gp_mp.BlendedGpMp(gp_mp.ProGP)
    mean_gmp, var_blended = gp_mp.predict_BlendedPos(test_x.reshape(-1,1))
    mean_gmp_arr = np.vstack(mean_gmp)   # (3, Nt) en mm

    # --- Crear GP para quaternions (si dispones de Q training) ---
    gp_quat = None
    mean_quat = None
    mean_quat_arr = None
    if use_csv_quats and Q is not None:
        gp_quat = ProGpMp(X, Q, via_t, vias_quat, dim=4, demos=demostraciones, size=Q.shape[0]//demostraciones, observation_noise=0.01)
        gp_quat.BlendedGpMp(gp_quat.ProGP)
        mean_quat, var_quat = gp_quat.predict_BlendedPos(test_x.reshape(-1,1))
        mean_quat_arr = np.vstack(mean_quat)   # (4, Nt)

    # --- inicializar GUI y PyBullet ---
    last_pts = vias.copy()   # mm
    print("Puntos inciales (mm):", last_pts)
    if mean_quat_arr is not None:
        last_quats = vias_quat.copy()
    else:
        last_quats = np.tile(np.array([0.0,0.0,0.0,1.0]), (vias.shape[0],1))
    

    dp3 = CameraOnlyPoints3D(last_pts, on_update=print_update_3d)
    ax = dp3.ax

    # dibujar prior + initial conditioning
    last_changed = [0]
    try:
        pathwise_update_fn()
    except Exception:
        pass

    plt.show(block=False)

    # iniciar PyBullet
    try:
        pb = PyBulletSpheres(use_gui=True)
        initial_positions_cm = (vias * 10.0).copy()
        pb.create_spheres(initial_positions_cm, radius_cm=5.0, rgba=[0.8,0.2,0.2,0.5])
        pb.load_kuka(base_pos=[0,0,0], urdf_rel_path="kuka_iiwa/model.urdf")
        pb.set_path(mean_gmp_arr.T * 10.0)
        pb.start_robot_follow(speed=100, ee_link=6)
    except Exception as exc:
        print("Error inicializando PyBullet:", exc)
        pb = None

    # Camera / ArUco params
    camera_matrix = np.array([
        [910.1155107777962, 0.0, 360.3277519024787],
        [0.0, 910.2233367566544, 372.6634999577232],
        [0.0, 0.0, 1.0]
    ])
    dist_coeffs = np.array([0.0212284835698144, 0.8546829039917951, 0.0034281408326615323, 0.0005749116561059772, -3.217248182814475])

    # mapping markers -> indices (ejemplo)
    marker_map = {66: 0, 24: 1}
    marker_lengths = {66: 0.053, 24: 0.063, "default": 0.053}

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

    try:
            dp3.show()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if 'dp3' in locals() and dp3 is not None:
                dp3.stop_camera(wait_timeout=5.0)
        except Exception:
            pass
        try:
            if pb is not None:
                pb.stop()
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
