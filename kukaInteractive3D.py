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

# --- Added imports for PyBullet and threading ---
import threading
import pybullet as p
import pybullet_data

# --------------------------
# Variables globales
# --------------------------
last_changed = None
last_pts = None

mean_gmp_arr = np.array([])
vias = np.array([])
via_t = np.array([])
last_final_line, prior_line = None, None
ax = None

pb = None  # handler PyBullet (se inicializa en main)
test_x = None  # grid temporal (se define en main)



# --------------------------
# GUI interactiva Matplotlib
# --------------------------
class DraggablePoints3D:
    def __init__(self, points, on_update=None):
        global ax
        self.points = np.array(points, float)
        self.last_points = self.points.copy()
        self.on_update = on_update

        self.fig = plt.figure(figsize=(8,8))
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self.ax.mouse_init(rotate_btn=3, zoom_btn=2)

        xs, ys, zs = self.points.T
        self.scatter = self.ax.scatter(xs, ys, zs, s=100, c='red', picker=5)

        self.texts = []
        for i,(x,y,z) in enumerate(self.points):
            txt = self.ax.text(x, y, z, str(i+1),
                            color='white', ha='center', va='center',
                            bbox=dict(boxstyle='circle', fc='blue', alpha=0.6))
            self.texts.append(txt)

        self.annot = self.ax.annotate('', xy=(0,0), xytext=(10,10),
                                    textcoords='offset points',
                                    bbox=dict(fc='w'),
                                    arrowprops=dict(arrowstyle='->'))
        self.annot.set_visible(False)

        c = self.fig.canvas
        c.mpl_connect('button_press_event',   self.on_press)
        c.mpl_connect('motion_notify_event',  self.on_motion)
        c.mpl_connect('button_release_event', self.on_release)
        c.mpl_connect('scroll_event',         self.on_scroll)

        self.press = None
        self.ind   = None
        self.ax.set_title('Drag XY, scroll to change Z')
        ax = self.ax

    def _notify(self):
        diffs = np.any(self.points != self.last_points, axis=1)
        changed = np.where(diffs)[0].tolist()
        idxs = changed if changed else None
        if self.on_update:
            self.on_update(self.points.copy(), idxs)
        self.last_points[:] = self.points

    def on_press(self, event):
        if event.button != 1 or event.inaxes != self.ax:
            return
        hit, info = self.scatter.contains(event)
        if not hit:
            return
        self.ind = info['ind'][0]
        x0,y0,z0 = self.points[self.ind]
        self.press = (x0, y0, z0, event.x, event.y)
        self.annot.xy = (event.x, event.y)
        self.annot.set_text(f'P{self.ind+1}: ({x0:.2f},{y0:.2f},{z0:.2f})')
        self.annot.set_visible(True)
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if not self.press or self.ind is None or event.inaxes != self.ax or event.button != 1:
            return
        x0,y0,z0, xp,yp = self.press
        dx_pix, dy_pix = event.x - xp, event.y - yp
        xlim = self.ax.get_xlim();  ylim = self.ax.get_ylim()
        w,h = self.ax.bbox.width, self.ax.bbox.height
        dx = dx_pix * (xlim[1]-xlim[0]) / max(1.0, w)
        dy = dy_pix * (ylim[1]-ylim[0]) / max(1.0, h)
        new = np.array([x0+dx, y0+dy, z0])
        self.points[self.ind] = new
        xs, ys, zs = self.points.T
        self.scatter._offsets3d = (xs, ys, zs)
        self.texts[self.ind].set_position((new[0], new[1]))
        self.texts[self.ind].set_3d_properties(new[2], zdir='z')
        self.annot.xy = (event.x, event.y)
        self.annot.set_text(f'P{self.ind+1}: ({new[0]:.2f},{new[1]:.2f},{new[2]:.2f})')
        self.fig.canvas.draw_idle()
        self._notify()

    def on_scroll(self, event):
        if self.ind is None or self.press is None:
            return
        step = event.step
        zlim = self.ax.get_zlim()
        dz = step * 0.02 * (zlim[1] - zlim[0])
        self.points[self.ind,2] += dz
        xs, ys, zs = self.points.T
        self.scatter._offsets3d = (xs, ys, zs)
        self.texts[self.ind].set_3d_properties(self.points[self.ind,2], zdir='z')
        xpix, ypix = event.x, event.y
        newz = self.points[self.ind,2]
        self.annot.xy = (xpix, ypix)
        self.annot.set_text(f'P{self.ind+1}: ({self.points[self.ind,0]:.2f},'
                            f'{self.points[self.ind,1]:.2f},{newz:.2f})')
        self.fig.canvas.draw_idle()
        self._notify()

    def on_release(self, event):
        self.press = None
        self.ind   = None
        self.annot.set_visible(False)
        self.fig.canvas.draw_idle()
        self._notify()

    def get_positions(self):
        return self.points.copy()

    def show(self):
        plt.show()



# --------------------------
# PyBullet handler (no bloqueante)
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

            #col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius_m, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius_m, rgbaColor=rgba, physicsClientId=self.client)

            for pos in positions_m:
                body = p.createMultiBody(baseMass=mass,
                                        #baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis,
                                        basePosition=[float(pos[0]), float(pos[1]), float(pos[2])],
                                        physicsClientId=self.client)
                self.bodies.append(body)
        return self.bodies

    def update_spheres(self, positions_cm):
        with self.lock:
            if not self.bodies:
                return
            positions_m = np.array(positions_cm, dtype=float) / 100.0
            if len(positions_m) != len(self.bodies):
                # recreate with bigger radius for visibility
                self.create_spheres(positions_cm, radius_cm=5.0, rgba=[0.8,0.2,0.2,0.5])
                return
            for bid, pos in zip(self.bodies, positions_m):
                p.resetBasePositionAndOrientation(bid, [float(pos[0]), float(pos[1]), float(pos[2])], [0,0,0,1], physicsClientId=self.client)

    # --- Robot ---
    def load_kuka(self, base_pos=[0,0,0], urdf_rel_path="kuka_iiwa/model.urdf"):
        with self.lock:
            kukaId = p.loadURDF(urdf_rel_path, base_pos, useFixedBase=True, physicsClientId=self.client)
            self.robot_id = kukaId
            # joints revolute
            numJoints = p.getNumJoints(kukaId, physicsClientId=self.client)
            self.robot_joints = [j for j in range(numJoints)
                                if p.getJointInfo(kukaId, j, physicsClientId=self.client)[2] == p.JOINT_REVOLUTE]
            # pose inicial
            # intentamos usar la pose rp original si tiene 7 valores; si no, ceros
            rp = [0, 0, 0, 0.5 * math.pi, 0, -0.5 * math.pi * 0.66, 0]
            for i, j in enumerate(self.robot_joints):
                try:
                    angle = rp[i] if i < len(rp) else 0.0
                    p.resetJointState(self.robot_id, j, angle, physicsClientId=self.client)
                except Exception:
                    pass

    def set_path(self, path_cm):
        """Establece el nuevo path (en cm). Thread-safe. Internamente guarda en metros."""
        with self.lock:
            if path_cm is None:
                self.path_m = None
            else:
                arr = np.array(path_cm, dtype=float)
                self.path_m = arr * 0.01  # cm -> m
                
            self.path_version += 1

    def draw_path(self, points_cm, step=5, color=[1,0,0], width=2.0, lifeTime=0):
        """Dibuja el path en PyBullet. Para ser rápido, dibuja saltando 'step' puntos."""
        with self.lock:
            # borrar path antiguo
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
            # muestreo
            n = pts.shape[0]
            if n < 2:
                return
            # convert cm->m
            pts_m = pts * 0.01
            # dibujar segmentos con salto 'step'
            for i in range(0, n-1, step):
                a = pts_m[i].tolist()
                b = pts_m[min(i+step, n-1)].tolist()
                try:
                    pid = p.addUserDebugLine(a, b, lineColorRGB=color, lineWidth=width, lifeTime=lifeTime, physicsClientId=self.client)
                    self.path_ids.append(pid)
                except Exception:
                    pass

    def start_robot_follow(self, speed=1.0, ee_link=None):
        """Inicia hilo que sigue self.path_m sin bloquear el main thread."""
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
        """Devuelve la posición del efector final en metros (np.array 3)."""
        try:
            st = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True, physicsClientId=self.client)
            # st puede tener diferentes longitudes segun versión; intentamos índices seguros
            # st[4] suele ser worldLinkFramePosition; st[0] es worldPosition.
            if len(st) >= 5 and st[4] is not None:
                return np.array(st[4], dtype=float)
            else:
                return np.array(st[0], dtype=float)
        except Exception:
            return None

    def _robot_loop(self, speed):
        """Loop que recorre la path actual; cuando la path cambia (path_version) se recalcula."""
        while self._robot_running:
            with self.lock:
                local_path = None if self.path_m is None else (self.path_m.copy())
                local_version = self.path_version

            if local_path is None or local_path.size == 0:
                time.sleep(0.05)
                continue

            # obtener posición actual del efector
            ee_pos = self._get_ee_position()
            if ee_pos is None:
                idx0 = 0
            else:
                dists = np.linalg.norm(local_path - ee_pos.reshape(1,3), axis=1)
                idx0 = int(np.argmin(dists))
            i = idx0
            # recorre desde idx0 hasta el final, pero si path_version cambia, sale del for externo para recalcular
            while i < len(local_path) and self._robot_running:
                with self.lock:
                    # si la path fue actualizada, salimos para recalcular
                    if local_version != self.path_version:
                        break
                    # copia actual (puede cambiar fuera del lock)
                    path_copy = self.path_m.copy() if self.path_m is not None else None

                if path_copy is None or path_copy.size == 0:
                    break

                target = path_copy[i].tolist()
                # calcular IK y mandar objetivo
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

                # sleep para simular velocidad (speed multiplica el timestep)
                time.sleep(self.time_step * max(0.01, speed))
                i += 1

            # breve espera antes de re-evaluar
            time.sleep(0.005)

    def stop(self):
        # parar robot thread y sim thread, desconectar PyBullet
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
# Callback integrado: actualiza pb y pathwise
# --------------------------
def print_update_3d(pts, changed):
    global last_changed, last_pts, pb
    last_changed = changed
    last_pts     = pts.copy()

    if pb is not None:
        pb.update_spheres(last_pts*10)

    if last_changed is not None:
        pathwise_update_fn()


def pathwise_update_fn():
    global mean_gmp_arr, vias, via_t, last_changed, last_pts, last_final_line, prior_line, ax, pb, test_x

    # kernel y GP-based pathwise update (igual que tu código original)
    BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
    base_kernels = [cls(lengthscales=1.0) for cls in BaseClasses]
    kernel = kernels.SeparateIndependent(base_kernels)

    # evitar sobreescribir via_t global: usamos via_t_local
    via_t_local = np.array([
        via_t[0],
        via_t[-1]/2,
        via_t[-1]
    ])

    X_obs = tf.convert_to_tensor(via_t_local, dtype=default_float())

    vias_local = last_pts.copy()
    Y_obs = tf.expand_dims(tf.convert_to_tensor(vias_local, dtype=default_float()), axis=0)

    # extraer del prior sus valores en esos puntos (obs_idx)
    obs_idx = [int(np.argmin(np.abs(test_x - t))) for t in via_t_local[:,0]]
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

    # actualizar matplotlib plot (global ax)
    try:
        global last_final_line, prior_line
        if last_final_line is not None:
            last_final_line.remove()
        if prior_line is not None:
            prior_line.remove()
        prior_line, = ax.plot(mean_gmp_arr[0], mean_gmp_arr[1], mean_gmp_arr[2], '--', color='blue', linewidth=2, label='GMP Prior')
        last_final_line, = ax.plot(mean_combo[:,0], mean_combo[:,1], mean_combo[:,2], '-.', color='red', linewidth=3, label='Final')
        ax.set_xlabel('x [mm]', fontsize=12)
        ax.set_ylabel('y [mm]', fontsize=12)
        ax.set_zlabel('z [mm]', fontsize=12)
        ax.legend(fontsize=12)
        plt.draw()
    except Exception:
        pass

    # actualizar el path del robot (no bloqueante)
    if pb is not None:
        # pb.set_path espera cm (coherente con tu uso anterior)
        pb.set_path(mean_combo * 10.0)   # mean_combo está en mm -> *10 => cm
        # dibujar path (muestreado)
        #pb.draw_path(mean_combo * 10.0, step=6, color=[1,0,0], width=2.0, lifeTime=0)


# --------------------------
# MAIN (igual flujo que tu script original)
# --------------------------
if __name__== '__main__':
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

    ruta_carpeta = '/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/RAIL/PRESSING'
    number = 6
    data = leer_archivos_mat(ruta_carpeta, number)

    dt = 0.01
    gap = 15

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

    # Construir training / puntos via
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

    var_blended[0] = np.where(var_blended[0]<0, 0, var_blended[0])
    var_blended[1] = np.where(var_blended[1]<0, 0, var_blended[1])
    var_blended[2] = np.where(var_blended[2]<0, 0, var_blended[2])

    # Visualizaciones Matplotlib (como antes)
    font_size = 25
    fig = plt.figure(figsize=(16, 8), dpi=100)
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.5, hspace=0.5, bottom=0.15, top=0.99)

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(vias[:, 0], vias[:, 1], vias[:, 2], s=600, c='blue', marker='x')
    ax1.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=20, c='blue', marker='o', alpha=0.3)
    ax1.plot(mean_gmp[0], mean_gmp[1], mean_gmp[2], c='black', linewidth=5, label='$ProGpMp$')
    ax1.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
    ax1.set_xlabel('$x$/mm', fontsize=font_size)
    ax1.set_ylabel('$y$/mm', fontsize=font_size)
    ax1.set_zlabel('$z$/mm', fontsize=font_size)

    ax2 = fig.add_subplot(322)
    ax2.plot(test_x, mean_gmp[0], c='red', linewidth=3, label='$x_{GMP}$')
    ax2.fill_between(test_x, mean_gmp[0] - 5 * np.sqrt(var_blended[0]), mean_gmp[0] + 5 * np.sqrt(var_blended[0]), color='red', alpha=0.3)
    ax2.scatter(via_t[:, 0], vias[:, 0], s=200, c='red', marker='x')
    ax2.scatter(X[:, 0], Y[:, 0], s=10, c='red', marker='o', alpha=0.3)
    ax2.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
    ax2.set_xlabel('(b)', fontsize=font_size)
    ax2.set_ylabel('$x$/mm', fontsize=font_size)

    ax3 = fig.add_subplot(324)
    ax3.plot(test_x, mean_gmp[1], c='blue', linewidth=3, label='$y_{GMP}$')
    ax3.fill_between(test_x, mean_gmp[1] - 5 * np.sqrt(var_blended[1]), mean_gmp[1] + 5 * np.sqrt(var_blended[1]), color='blue', alpha=0.3)
    ax3.scatter(via_t[:, 0], vias[:, 1], s=200, c='blue', marker='x')
    ax3.scatter(X[:, 0], Y[:, 1], s=10, c='blue', marker='o', alpha=0.3)
    ax3.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
    ax3.set_xlabel('(c)', fontsize=font_size)
    ax3.set_ylabel('$y$/mm', fontsize=font_size)

    ax4 = fig.add_subplot(326)
    ax4.plot(test_x, mean_gmp[2], c='green', linewidth=3, label='$z_{GMP}$')
    ax4.fill_between(test_x, mean_gmp[2] - 5 * np.sqrt(var_blended[2]), mean_gmp[2] + 5 * np.sqrt(var_blended[2]), color='green', alpha=0.3)
    ax4.scatter(via_t[:, 0], vias[:, 2], s=200, c='green', marker='x')
    ax4.scatter(X[:, 0], Y[:, 2], s=10, c='green', marker='o', alpha=0.3)
    ax4.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
    ax4.set_xlabel('(d)', fontsize=font_size)
    ax4.set_ylabel('$z$/mm', fontsize=font_size)

    plt.show(block=False)

    old_mean = mean_gmp.copy()
    vias = np.array([ vias[0], vias[1]/2, vias[1] ])
    mean_gmp_arr = np.vstack(mean_gmp)   # shape (3, N)

    try:
        pb = PyBulletSpheres(use_gui=True)
        pb.create_spheres(vias*10, radius_cm=5.0, rgba=[0.8,0.2,0.2,0.5])
        pb.load_kuka(base_pos=[0,0,0], urdf_rel_path="kuka_iiwa/model.urdf")
        pb.set_path(mean_gmp_arr.T*10)   # mean_gmp_arr en mm -> *10 => cm
        pb.start_robot_follow(speed=20, ee_link=6)

    except Exception as exc:
        print("Error inicializando PyBullet / KUKA:", exc)
        pb = None

    # GUI interactiva Matplotlib (bloquea aquí pero robot sigue en su hilo)
    dp3 = DraggablePoints3D(vias, on_update=print_update_3d)
    try:
        dp3.show()  # aquí se mantiene la GUI; el robot en background seguirá siguiendo path
    except KeyboardInterrupt:
        pass
    finally:
        if pb is not None:
            pb.stop()

    final_positions = dp3.get_positions()
    print("Final 3D (cm):", final_positions)
