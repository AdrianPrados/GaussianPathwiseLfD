import numpy as np
import matplotlib.pyplot as plt

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


#Global variables para almacenar el estado
last_changed = None

last_pts = None

mean_gmp_arr =np.array([])
vias = np.array([])
via_t = np.array([])
last_final_line, prior_line = None, None
ax = None

class DraggablePoints3D:
    def __init__(self, points, on_update=None):
        """
        points: array Nx3 de posiciones iniciales (x,y,z).
        on_update: callback que recibe (pts_current:Nx3, changed_idxs:list or None)
                en cada movimiento.
        """
        global ax
        self.points = np.array(points, float)   # Nx3
        self.last_points = self.points.copy()
        self.on_update = on_update

        # crear figura y eje 3D
        self.fig = plt.figure(figsize=(8,8))
        self.ax  = self.fig.add_subplot(111, projection='3d')
        # dejamos activa la rotación con botón derecho, zoom con rueda
        self.ax.mouse_init(rotate_btn=3, zoom_btn=2)

        # dibujar scatter
        xs, ys, zs = self.points.T
        self.scatter = self.ax.scatter(xs, ys, zs, s=100, c='red', picker=5)

        # etiquetas numeradas
        self.texts = []
        for i,(x,y,z) in enumerate(self.points):
            txt = self.ax.text(x, y, z, str(i+1),
                            color='white', ha='center', va='center',
                            bbox=dict(boxstyle='circle', fc='blue', alpha=0.6))
            self.texts.append(txt)

        # anotación 2D
        self.annot = self.ax.annotate('', xy=(0,0), xytext=(10,10),
                                    textcoords='offset points',
                                    bbox=dict(fc='w'),
                                    arrowprops=dict(arrowstyle='->'))
        self.annot.set_visible(False)

        # conectar eventos
        c = self.fig.canvas
        c.mpl_connect('button_press_event',   self.on_press)
        c.mpl_connect('motion_notify_event',  self.on_motion)
        c.mpl_connect('button_release_event', self.on_release)
        c.mpl_connect('scroll_event',         self.on_scroll)

        self.press = None  # almacenará (x0,y0,z0, xpix, ypix)
        self.ind   = None
        self.ax.set_title('Drag the vias (XY), spinning wheel (Z)')
        
        ax = self.ax  # guardar referencia global para uso posterior

    def _notify(self):
        diffs = np.any(self.points != self.last_points, axis=1)
        changed = np.where(diffs)[0].tolist()
        idxs = changed if changed else None
        if self.on_update:
            self.on_update(self.points.copy(), idxs)
        self.last_points[:] = self.points

    def on_press(self, event):
        # solo botón izquierdo sobre puntos
        if event.button != 1 or event.inaxes != self.ax:
            return
        hit, info = self.scatter.contains(event)
        if not hit:
            return
        self.ind = info['ind'][0]
        x0,y0,z0 = self.points[self.ind]
        # guardamos posición y pixel actual
        self.press = (x0, y0, z0, event.x, event.y)
        # anotación en pantalla
        self.annot.xy = (event.x, event.y)
        self.annot.set_text(f'P{self.ind+1}: ({x0:.2f},{y0:.2f},{z0:.2f})')
        self.annot.set_visible(True)
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if not self.press or self.ind is None or event.inaxes != self.ax or event.button != 1:
            return
        x0,y0,z0, xp,yp = self.press
        dx_pix, dy_pix = event.x - xp, event.y - yp
        # convertir píxeles a datos X,Y
        xlim = self.ax.get_xlim();  ylim = self.ax.get_ylim()
        w,h = self.ax.bbox.width, self.ax.bbox.height
        dx = dx_pix * (xlim[1]-xlim[0]) / w
        dy = dy_pix * (ylim[1]-ylim[0]) / h
        new = np.array([x0+dx, y0+dy, z0])
        self.points[self.ind] = new
        # actualizar scatter3D
        xs, ys, zs = self.points.T
        self.scatter._offsets3d = (xs, ys, zs)
        # actualizar etiquetas
        self.texts[self.ind].set_position((new[0], new[1]))
        self.texts[self.ind].set_3d_properties(new[2], zdir='z')
        # actualizar anotación
        self.annot.xy = (event.x, event.y)
        self.annot.set_text(f'P{self.ind+1}: ({new[0]:.2f},{new[1]:.2f},{new[2]:.2f})')
        self.fig.canvas.draw_idle()
        self._notify()

    def on_scroll(self, event):
        # modificar Z solo si estamos arrastrando (botón izquierdo presionado)
        if self.ind is None or self.press is None:
            return
        # event.step es +1 (hacia arriba) o -1 (hacia abajo)
        step = event.step
        # definimos un incremento relativo al rango actual de Z
        zlim = self.ax.get_zlim()
        dz = step * 0.02 * (zlim[1] - zlim[0])  # 2% por paso de rueda
        self.points[self.ind,2] += dz
        # actualizar scatter y etiquetas
        xs, ys, zs = self.points.T
        self.scatter._offsets3d = (xs, ys, zs)
        self.texts[self.ind].set_3d_properties(self.points[self.ind,2], zdir='z')
        # actualizar anotación (en píxeles sigue igual)
        xpix, ypix = event.x, event.y
        newz = self.points[self.ind,2]
        self.annot.xy = (xpix, ypix)
        self.annot.set_text(f'P{self.ind+1}: ({self.points[self.ind,0]:.2f},'
                            f'{self.points[self.ind,1]:.2f},{newz:.2f})')
        self.fig.canvas.draw_idle()
        self._notify()

    def on_release(self, event):
        # fin de arrastre
        self.press = None
        self.ind   = None
        self.annot.set_visible(False)
        self.fig.canvas.draw_idle()
        self._notify()

    def get_positions(self):
        """Devuelve un array Nx3 con las posiciones actuales."""
        return self.points.copy()

    def show(self):
        plt.show()
        
    # callback ejemplo
def print_update_3d(pts, changed):
    global last_changed, last_pts
    last_changed = changed     
    last_pts     = pts.copy()
    
    if last_changed is not None:
        pathwise_update_fn() 
    else:
        print("No hubo cambios en las posiciones.")
        
def pathwise_update_fn():
    global mean_gmp_arr, vias, via_t, last_changed, last_pts, last_final_line, prior_line, ax
    
    #* Pathwise optimization in 3D
    BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential, kernels.SquaredExponential)
    base_kernels = [cls(lengthscales=1.0) for cls in BaseClasses]
    kernel = kernels.SeparateIndependent(base_kernels)

    F_prior_full = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()),axis=0)

    via_t = np.array([
        via_t[0],
        via_t[-1]/2,   
        via_t[-1]    
    ])

    X_obs = tf.convert_to_tensor(via_t, dtype=default_float())

    vias = last_pts.copy()

    Y_obs = tf.expand_dims(tf.convert_to_tensor(vias, dtype=default_float()), axis=0)

    # 3) Extraemos del prior sus valores en esos puntos
    obs_idx = [int(np.argmin(np.abs(test_x - t))) for t in via_t[:,0]]
    F_obs = tf.expand_dims(
        tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()),
        axis=0
    )  # shape [1, N_obs, 2]

    dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)

    # 5) Muestreamos la corrección en toda la rejilla
    X_test = test_x.reshape(-1,1)
    X_test = tf.convert_to_tensor(test_x.reshape(-1, 1), dtype=default_float())  # (763, 1)
    #X_test = tf.expand_dims(X_test, axis=0)  # (1, 763, 1)
    dF_full = dF_fn(X_test)
    F_cond = F_prior_full + dF_full

    mean_corrected = F_cond.numpy()
    #mean_gp_arr = np.squeeze(mean_gmp_arr)
    mean_combo = np.squeeze(mean_corrected)

    if last_final_line is not None:
        last_final_line.remove()
        
    if prior_line is not None:
        prior_line.remove()


    prior_line,= ax.plot(mean_gmp_arr[0], mean_gmp_arr[1], mean_gmp_arr[2],'--', color='blue', linewidth=2, label='GMP Prior')
    last_final_line, = ax.plot(mean_combo[:,0], mean_combo[:,1], mean_combo[:,2],'-.',color='red', linewidth=3, label='Final')

    ax.set_xlabel('x [mm]', fontsize=12)
    ax.set_ylabel('y [mm]', fontsize=12)
    ax.set_zlabel('z [mm]', fontsize=12)
    ax.legend(fontsize=12)
    #plt.tight_layout()
    #plt.show()

if __name__== '__main__':
    #! ---- Code for real data from robots----
    """ dt = 0.01
    gap = 30
    demostraciones = 1
    data_dict = {}
    values = []
    lengths = []

    for i in range(1, demostraciones + 1):
        file_name = f"/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/ExpCartesianADAM/obstacle_{i}.csv"
        data = pd.read_csv(file_name)
        lengths.append(len(data['x'][::gap]))

    min_length = min(lengths)

    for i in range(1, demostraciones + 1):
        file_name = f"/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/ExpCartesianADAM/obstacle_{i}.csv"
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
            Y = np.vstack((Y, vias)) """
    #!---- Code for RAIL datatset ----
    def leer_archivos_mat(ruta_carpeta,number):
        data = {}
        for archivo in os.listdir(ruta_carpeta):
            if archivo.endswith(str(number)+'.mat'):
                # Extract name
                nombre_archivo, extension = os.path.splitext(archivo)
                try:
                    clave = int(nombre_archivo)
                except ValueError:
                    continue  # Ignore archives withou numeric number
                
                # Complete path
                ruta_archivo = os.path.join(ruta_carpeta, archivo)
                
                # Load data from .mat
                datos = loadmat(ruta_archivo)
                
                # Save dta in dict
                data[clave] = datos

        return data



    # Path to folder with .mat
    ruta_carpeta = '/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/RAIL/PRESSING'
    number = 6
    data = leer_archivos_mat(ruta_carpeta,number)

    # Example of use: print the data from 1.mat
    #print(data[1]['dataset']['pos'][0][0].T)  #data[1]['dataset']['pos'][numero de demostraciones][0].T

    dt = 0.01
    gap = 30

    #*Loading all the data
    demostraciones = 1
    demos = data[number]['dataset']
    for i in range(demostraciones):
        demo = demos[i]
        pos = demo['pos'][0].T[:, 0::gap]*10  # np.ndarray, shape: (3,1000/gap)
        vel = demo['vel'][0].T[:, 0::gap]  # np.ndarray, shape: (3,1000/gap)
        acc = demo['acc'][0].T[:, 0::gap]  # np.ndarray, shape: (3,1000/gap)
        t = demo['time'][0].T[:, 0::gap]  # np.ndarray, shape: (1,1000/gap)
        print ("Tiempo: ",type(t))
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
    font_size = 18
    print(type(pos))
    #* Selecting the target position and time and via points
    #! Points for Real data (constructed by hand, you can use what ever you want)
    """ target_t = t[0][-1]
    target_position=np.array([values[2][0][-1], values[2][1][-1], values[2][2][-1]])

    via_point0_t = t[0][0]
    via_point0_position= np.array([x_data[0], y_data[0], z_data[0]]) """

    """ via_point1_t = t[0][106]
    via_point1_position = np.array([values[2][0][106],values[2][1][106], values[2][2][106]]) """

    #* Constructing the training set with the target and via points
    #! Points for RAIL data
    #size = demos[0].pos[:, 0::gap].T.shape[0]
    target_t = sum(demos[i]['time'][0].T[:, 0::gap][0, -1] for i in range(demostraciones)) / demostraciones
    target_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, -1]*10 for i in range(demostraciones)) / demostraciones
    via_point0_t = sum(demos[i]['time'][0].T[:, 0::gap][0, 0] for i in range(demostraciones)) / demostraciones
    via_point0_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, 0]*10 for i in range(demostraciones)) / demostraciones
    via_point1_t = sum(demos[i]['time'][0].T[:, 0::gap][0, demos[i]['pos'][0].T[:, 0::gap].shape[1] * 2 // 10] for i in range(demostraciones)) / demostraciones
    via_point1_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, demos[i]['pos'][0].T[:, 0::gap].shape[1] * 2 // 10]*10 for i in range(demostraciones)) / demostraciones
    #Vias points

    via_t = np.array([via_point0_t, target_t]).reshape(-1, 1)
    vias = np.array([via_point0_position, target_position])


    #time.sleep(1000)
    # predicting for dim0   --> size=demos[0]['pos'][0].T[:, 0::gap].T.shape[0]
    observation_noise = 0.4
    #gp_mp= ProGpMp(X, Y, via_t, vias,dim=3, demos=demostraciones,size = demos[0].pos[:, 0::gap].T.shape[0] , observation_noise=observation_noise) # For RAIL data use: demos[0].pos[:, 0::gap].T.shape[0]
    gp_mp= ProGpMp(X, Y, via_t, vias,dim=3, demos=demostraciones,size = size, observation_noise=observation_noise) # For real data

    gp_mp.BlendedGpMp(gp_mp.ProGP) #? If you use more than one GpMp is mandatory to use BlendedGpMp, input: list[]
    test_x = np.arange(0.0, target_t, dt)
    #test_x = np.arange(vias[0,0],vias[2,0],(1/pos0.size))


    #alpha_list = (np.tanh((test_x - 0.5) * 5) + 1.0) / 2
    #print(type(alpha_list))
    alpha_list=np.ones(len(test_x))
    alpha_list = np.vstack((alpha_list, alpha_list))
    alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))

    # alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))
    mean_gmp, var_blended = gp_mp.predict_BlendedPos(test_x.reshape(-1, 1))

    #time.sleep(1000)
    var_blended[0]=np.where(var_blended[0]<0,0,var_blended[0])
    var_blended[1]=np.where(var_blended[1]<0,0,var_blended[1])
    var_blended[2]=np.where(var_blended[2]<0,0,var_blended[2])


    font_size = 25
    fig = plt.figure(figsize=(16, 8), dpi=100)
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.5, hspace=0.5, bottom=0.15, top=0.99)
    #* Visualization in 3D

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(vias[:, 0], vias[:, 1], vias[:, 2], s=600, c='blue', marker='x')
    ax1.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=20, c='blue', marker='o', alpha=0.3)
    ax1.plot(mean_gmp[0], mean_gmp[1], mean_gmp[2], c='black', linewidth=5, label='$GMP$')

    ax1.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
    ax1.set_xlabel('$x$/mm', fontsize=font_size)
    ax1.set_ylabel('$y$/mm', fontsize=font_size)
    ax1.set_zlabel('$z$/mm', fontsize=font_size)

    #* 2D visualization
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

    plt.show()
    old_mean = []
    old_mean = mean_gmp.copy()
    # crear puntos iniciales Nx3
    vias = np.array([
        vias[0],  
        vias[1]/2,
        vias[1]
    ])
    
    mean_gmp_arr = np.vstack(mean_gmp)

    dp3 = DraggablePoints3D(vias, on_update=print_update_3d)
    dp3.show()

    # luego puedes obtener
    final_positions = dp3.get_positions()
    print("Final 3D:", final_positions)

