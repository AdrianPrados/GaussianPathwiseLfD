import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# GPflow / ProGP imports
import tensorflow as tf
from scipy.io import loadmat
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import updates
from ProGP import ProGpMp

import pybullet as p
import pybullet_data


try:
    from Adam_sim.scripts.adam import ADAM
    ADAM_AVAILABLE = True
except Exception as e:
    print("Warning: scripts.adam import failed:", e)
    ADAM_AVAILABLE = False

ADAM_URDF_PATH = "/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/Adam_sim/models/robot/rb1_base_description/robots/robotDummy.urdf" #! change to your path to URDF file for ADAM urdf

# ---------- Globals ----------
last_changed = None
last_pts = None
mean_gmp_arr = np.array([])
vias = np.array([])
via_t = np.array([])
last_final_line, prior_line = None, None
ax = None
test_x = None

# ---------- DraggablePoints3D (igual comportamiento) ----------
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

# ---------- PyBulletManager (combina ADAM start + PyBullet spheres) ----------
class PyBulletManager:
    def __init__(self, start_adam=False, adam_urdf=None, wait_adam_timeout=8.0, gravity=-9.81, time_step=1./240.):
        """
        start_adam: si True e ADAM_AVAILABLE, arrancará ADAM en background.
        adam_urdf: ruta que se pasará a ADAM si se arranca.
        wait_adam_timeout: tiempo máximo a esperar a que ADAM esté listo para intentar extraer su client id.
        """
        self.client = None
        self.time_step = time_step
        self._sim_thread = None
        self._sim_running = False
        self.bodies = []
        self.lock = threading.Lock()
        self.path_ids = []
        self.path_m = None
        self.path_version = 0

        # ADAM related
        self.adam = None
        self.adam_ready = False
        self.adam_exc = None
        self._adam_thread = None
        self._start_adam = start_adam and ADAM_AVAILABLE
        self._adam_urdf = adam_urdf if adam_urdf is not None else ADAM_URDF_PATH
        self.wait_adam_timeout = wait_adam_timeout

        # if start_adam requested and ADAM available, start in background
        if self._start_adam:
            self._adam_thread = threading.Thread(target=self._init_adam, daemon=True)
            self._adam_thread.start()
            # wait a short time for ADAM to initialize and expose client; we will attempt to read client id
            waited = 0.0
            poll = 0.15
            while waited < self.wait_adam_timeout:
                if self.adam_ready:
                    break
                if self.adam_exc is not None:
                    break
                time.sleep(poll)
                waited += poll

        # now decide which physics client to use:
        cid = None
        if self._start_adam and self.adam_ready:
            cid = self._extract_adam_client()
            if cid is not None:
                print("PyBulletManager: usando physicsClientId expuesto por ADAM:", cid)
            else:
                print("PyBulletManager: ADAM arrancado pero no expone physicsClientId; usaremos cliente DIRECT interno.")
        else:
            if self._start_adam and not ADAM_AVAILABLE:
                print("PyBulletManager: ADAM no disponible en este entorno; creando cliente DIRECT.")

        # fallback: create DIRECT client if no ADAM client found
        if cid is None:
            try:
                self.client = p.connect(p.DIRECT)
            except Exception as e:
                raise RuntimeError("No se pudo conectar a PyBullet DIRECT: " + str(e))
        else:
            self.client = cid

        # configure client
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0,0,gravity, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)
        try:
            _ = p.loadURDF("plane.urdf", physicsClientId=self.client)
        except Exception:
            pass

        # start local stepSimulation thread for the client we are using
        self._sim_running = True
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

    # ---------------- ADAM init ----------------
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
        """
        Intenta localizar un atributo entero plausible en la instancia ADAM que represente el physics client id.
        """
        if self.adam is None:
            return None
        cand_names = ['physicsClientId', 'physics_client', 'client', 'pybullet_client', 'pb_client',
                    '_physics_client', 'physicsClient', 'client_id', '_client', 'pclient']
        for name in cand_names:
            if hasattr(self.adam, name):
                v = getattr(self.adam, name)
                # Si es int y plausible -> OK
                if isinstance(v, int) and v >= 0:
                    return v
                # si es wrapper object con attr 'client'
                if hasattr(v, 'client'):
                    c = getattr(v, 'client')
                    if isinstance(c, int) and c >= 0:
                        return c
        # fallback: inspeccionar __dict__
        for k, val in getattr(self.adam, '__dict__', {}).items():
            if isinstance(val, int) and val >= 0 and val < 100:
                return val
        return None

    # ---------------- simulation loop ----------------
    def _sim_loop(self):
        while self._sim_running:
            with self.lock:
                try:
                    p.stepSimulation(physicsClientId=self.client)
                except Exception:
                    # si ADAM está gestionando su propio stepping, es posible que p.stepSimulation falle; ignoramos
                    pass
            time.sleep(self.time_step)

    # ---------------- sphere helpers ----------------
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
                b = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[float(pos[0]),float(pos[1]),float(pos[2])], physicsClientId=self.client)
                self.bodies.append(b)
        return self.bodies

    def update_spheres(self, positions_cm):
        with self.lock:
            if not self.bodies: return
            positions_m = np.array(positions_cm)/100.0
            if len(positions_m) != len(self.bodies):
                # recreate at larger radius for visibility
                self.create_spheres(positions_cm, radius_cm=5.0, rgba=[0.8,0.2,0.2,0.5])
                return
            for bid,pos in zip(self.bodies, positions_m):
                try:
                    p.resetBasePositionAndOrientation(bid, [float(pos[0]),float(pos[1]),float(pos[2])], [0,0,0,1], physicsClientId=self.client)
                except Exception:
                    pass

    def set_path(self, path_cm):
        """Guardamos path en metros internamente (no dibujado aquí)."""
        with self.lock:
            if path_cm is None:
                self.path_m = None
            else:
                arr = np.array(path_cm, dtype=float)
                self.path_m = arr * 0.01
            self.path_version += 1

    def stop(self, disconnect_client=False):
        # stop simulation thread
        self._sim_running = False
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=1.0)
            self._sim_thread = None
        # remove bodies (best-effort)
        with self.lock:
            for b in self.bodies:
                try: p.removeBody(b, physicsClientId=self.client)
                except Exception: pass
            self.bodies = []
        # optionally disconnect client if we own it (only if it is DIRECT fallback)
        if disconnect_client:
            try:
                p.disconnect(self.client)
            except Exception:
                pass
    def start_adam_follow(self, speed=1.0, ee_side='left', ee_link_name='hand'):
        # guarda parámetros
        self._follow_speed = float(speed)
        self._initialized = True
        self._ee_side = ee_side
        self._ee_link_name = ee_link_name
        self._robot_running = True
        self._robot_thread = threading.Thread(target=self._adam_robot_loop, daemon=True)
        self._robot_thread.start()

    def stop_adam_follow(self):
        """Para el hilo de seguimiento."""
        self._robot_running = False
        if getattr(self, '_robot_thread', None) is not None:
            self._robot_thread.join(timeout=1.0)
            self._robot_thread = None

    def _adam_robot_loop(self):
        time.sleep(10)
        # intentamos recuperar una orientación "por defecto" a partir de la pose actual del efector
        default_quat = [0, 1, 1, 1]
        try:
            cur = None
            try:
                cur = self.adam.arm_kinematics.get_arm_link_pose(self._ee_side, target_link=self._ee_link_name)
            except Exception:
                # algunos wrappers usan distinto nombre; lo ignoramos si falla
                cur = None
            if cur is not None and len(cur) >= 2 and cur[1] is not None:
                default_quat = list(cur[1])
        except Exception:
            default_quat = [1, 1, 1, 1]

        
        while self._robot_running:
            self.adam.hand_kinematics.move_hand_to_dofs('left', [1000, 1000, 1000, 1000, 1000, 1000])
            with self.lock:
                """ local_path = None if getattr(self, 'path_m', None) is None else (self.path_m.copy())
                local_version = getattr(self, 'path_version', 0) """
                local_path = None if self.path_m is None else (self.path_m.copy())
                local_version = self.path_version

            if local_path is None or local_path.size == 0:
                # damos un paso a ADAM (si procede) para mantener la sim viva
                try:
                    if self.ready and self.adam is not None:
                        # ADAM puede requerir que se llame step() para aplicar comandos / avanzar la física
                        self.adam.step()
                except Exception:
                    pass
                #time.sleep(max(0.02, getattr(self, 'point_dt', 0.03)))
                continue
            #self.adam.step()
            

            # obtener posición actual del efector desde ADAM
            cur = self.adam.arm_kinematics.get_arm_link_pose(self._ee_side, target_link=self._ee_link_name)
            default_quat = [0.16650719940662384, 0.26198387145996094, -0.7870264053344727, 0.5331315398216248]
            cur_pos = np.array(cur[0], dtype=float)
            if self._initialized == True:
                self._initialized = False
                index = 0
            """ else:
                dists = np.linalg.norm(local_path - cur_pos.reshape(1,3), axis=1)
                idx0 = int(np.argmin(dists))
                index = idx0 """
            
            while index < local_path.shape[0] and getattr(self, '_robot_running', False):
                with self.lock:
                    if local_version != getattr(self, 'path_version', local_version):
                        break
                    #path_copy = None if getattr(self, 'path_m', None) is None else (self.path_m.copy())
                    path_copy = self.path_m.copy() if self.path_m is not None else None

                if path_copy is None or path_copy.size == 0:
                    break
                target_pos = path_copy[index].tolist()   # target en metros [x,y,z]
                # Construir pose: [position, quaternion]
                #pose = [target_pos, default_quat]

                with self.lock:
                    #self.adam.arm_kinematics.move_arm_to_pose(arm='left',target_pose=pose, target_link='hand',pos_act=cur) #! No se por que es bloqueante :(
                    ee_index = self.adam.arm_kinematics.get_arm_link_index("left", "dummy")
                    joint_indices, rev_joint_indices = self.adam.arm_kinematics.get_arm_joint_indices("left")
                    ik_solution = p.calculateInverseKinematics(self.adam.robot_id, ee_index, target_pos, default_quat,
                                                    solver=0,
                                                    maxNumIterations=1000,
                                                    residualThreshold=.01)
                    arm_solution = [ik_solution[i] for i in rev_joint_indices]
                    arm_solution = self.adam.arm_kinematics.compute_closest_joints("left", arm_solution)
                    for i, joint_id in enumerate(joint_indices):
                        p.setJointMotorControl2(self.adam.robot_id, joint_id, p.POSITION_CONTROL, arm_solution[i])
                    #self.adam.wait(0.2)
                    self.adam.step()

                #dt_sleep = getattr(self, 'point_dt', 0.03) * max(0.01, float(getattr(self, '_follow_speed', 1.0)))
                
                #time.sleep(0.4)
                self.adam.wait(0.1)
                index= index + 1
                #print("Valor de i 2:", i)

            # breve pausa antes de re-evaluar path_version y posición actual
            time.sleep(0.005)


def print_update_3d(pts, changed):
    global last_changed, last_pts, pb_global
    last_changed = changed
    last_pts = pts.copy()
    if pb_global is not None:
        try:
            pb_global.update_spheres(last_pts*10)
        except Exception:
            pass
    if last_changed is not None:
        pathwise_update_fn()

# ---------- pathwise_update_fn ----------
def pathwise_update_fn():
    global mean_gmp_arr, vias, via_t, last_changed, last_pts, last_final_line, prior_line, ax, pb_global, test_x

    if mean_gmp_arr is None or mean_gmp_arr.size == 0 or test_x is None or last_pts is None:
        return

    BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
    base_kernels = [cls(lengthscales=1.0) for cls in BaseClasses]
    kernel = kernels.SeparateIndependent(base_kernels)

    via_t_local = np.array([
        via_t[0],
        via_t[-1]/2,
        via_t[-1]
    ])

    X_obs = tf.convert_to_tensor(via_t_local, dtype=default_float())
    vias_local = last_pts.copy()
    Y_obs = tf.expand_dims(tf.convert_to_tensor(vias_local, dtype=default_float()), axis=0)

    obs_idx = [int(np.argmin(np.abs(test_x - t))) for t in via_t_local[:,0]]
    F_obs = tf.expand_dims(
        tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()),
        axis=0
    )

    try:
        dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)
        X_test = tf.convert_to_tensor(test_x.reshape(-1, 1), dtype=default_float())
        dF_full = dF_fn(X_test)
        F_cond = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()), axis=0) + dF_full
        mean_corrected = F_cond.numpy()
        mean_combo = np.squeeze(mean_corrected)
    except Exception as exc:
        #print("pathwise_update_fn: fallo en updates.exact o TF:", exc)
        mean_combo = mean_gmp_arr.T.copy()

    
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

    # If pb_global exists, set its path (cm) so any robot follower could use it
    try:
        if pb_global is not None:
            pb_global.set_path(mean_combo * 10.0)
    except Exception:
        pass


if __name__ == "__main__":
    #! Load real demonstrations
    #! ---- Code for real data from robots----
    dt = 0.01
    gap = 30
    demostraciones = 1
    data_dict = {}
    values = []
    lengths = []

    for i in range(1, demostraciones + 1):
        file_name = f"/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/ExpCartesianADAM/pick2.csv"
        data = pd.read_csv(file_name)
        lengths.append(len(data['x'][::gap]))

    min_length = min(lengths)

    for i in range(1, demostraciones + 1):
        file_name = f"/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/ExpCartesianADAM/pick2.csv"
        data = pd.read_csv(file_name)
        
        x_data = np.array(data['x'][::gap])[:min_length]*10
        y_data = np.array(data['y'][::gap])[:min_length]*10
        z_data = np.array(data['z'][::gap])[:min_length]*10
        
        data_dict[f"obstacle_{i}"] = np.array([x_data, y_data, z_data])
        values.append(np.array([x_data, y_data, z_data]))


    for key, value in data_dict.items():
        pos = value
        t = np.linspace(0, 6, min_length).reshape(1, min_length)
        via_t = t.T
        vias = pos.T
        print(vias.shape)
        if key == "obstacle_1":
            size = vias.shape[0]
            X = via_t
            Y = vias
        else:
            X = np.vstack((X, via_t))
            Y = np.vstack((Y, vias))

    np.random.seed(30)

    #* Selecting the target position and time and via points
    #! Points for Real data (constructed by hand, you can use what ever you want)
    target_t = t[0][-1]
    target_position=np.array([values[0][0][-1], values[0][1][-1], values[0][2][-1]])

    via_point0_t = t[0][0]
    via_point0_position= np.array([x_data[0], y_data[0], z_data[0]])

    via_point1_t = t[0][16]
    via_point1_position = np.array([values[0][0][16],values[0][1][16], values[0][2][16]])

    via_t = np.array([via_point0_t,via_point1_t, target_t]).reshape(-1, 1)
    vias = np.array([via_point0_position,via_point1_position, target_position])

    observation_noise = 1.0
    gp_mp = ProGpMp(X, Y, via_t, vias, dim=3, demos=demostraciones, size=size, observation_noise=observation_noise)
    gp_mp.BlendedGpMp(gp_mp.ProGP)
    test_x = np.arange(0.0, target_t, dt)

    mean_gmp, var_blended = gp_mp.predict_BlendedPos(test_x.reshape(-1, 1))
    mean_gmp_arr = np.vstack(mean_gmp)

    # plotting
    """ font_size = 25
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

    ax3 = fig.add_subplot(324)
    ax3.plot(test_x, mean_gmp[1], c='blue', linewidth=3, label='$y_{GMP}$')
    ax3.fill_between(test_x, mean_gmp[1] - 5 * np.sqrt(var_blended[1]), mean_gmp[1] + 5 * np.sqrt(var_blended[1]), color='blue', alpha=0.3)
    ax3.scatter(via_t[:, 0], vias[:, 1], s=200, c='blue', marker='x')
    ax3.scatter(X[:, 0], Y[:, 1], s=10, c='blue', marker='o', alpha=0.3)

    ax4 = fig.add_subplot(326)
    ax4.plot(test_x, mean_gmp[2], c='green', linewidth=3, label='$z_{GMP}$')
    ax4.fill_between(test_x, mean_gmp[2] - 5 * np.sqrt(var_blended[2]), mean_gmp[2] + 5 * np.sqrt(var_blended[2]), color='green', alpha=0.3)
    ax4.scatter(via_t[:, 0], vias[:, 2], s=200, c='green', marker='x')
    ax4.scatter(X[:, 0], Y[:, 2], s=10, c='green', marker='o', alpha=0.3) """

    plt.show(block=False)
    
    old_mean = mean_gmp.copy()
    vias = np.array([ vias[0], vias[1], vias[-1] ])
    mean_gmp_arr = np.vstack(mean_gmp)   # shape (3, N)

    # --- Aquí usamos la nueva única clase PyBulletManager ---
    # start_adam=True intentará arrancar ADAM si está disponible y usar su GUI
    try:
        pb_global = PyBulletManager(start_adam=True, adam_urdf=ADAM_URDF_PATH)
    except Exception as exc:
        print("Error inicializando PyBulletManager:", exc)
        pb_global = None

    # crear esferas (si pb_global válido)
    if pb_global is not None:
        try:
            pb_global.create_spheres(vias*10, radius_cm=5.0, rgba=[0.8,0.2,0.2,0.5])
            pb_global.set_path(mean_gmp_arr.T*10)
            pb_global.start_adam_follow(speed=20.0, ee_side='left', ee_link_name='dummy')
            #pb_global.start_robot_follow(speed=20, ee_link=6)
        except Exception:
            pass

    # GUI interactiva Matplotlib (bloquea aquí; pb_global mantiene la sim en background)
    dp3 = DraggablePoints3D(vias, on_update=print_update_3d)
    try:
        dp3.show()
    except KeyboardInterrupt:
        pass
    finally:
        if pb_global is not None:
            pb_global.stop(disconnect_client=True)
        # si ADAM fue arrancado, dejamos su GUI en paz (no la cerramos aquí)

    final_positions = dp3.get_positions()
    print("Final 3D (cm):", final_positions)
