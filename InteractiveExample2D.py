import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# GPflowSampling imports
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import priors, updates, decoupled
from gpflow_sampling.sampling.core import AbstractSampler

# Original imports
import pyLasaDataset as lasa
from ProGP import ProGpMp
import time

#Global variables para almacenar el estado
last_changed = None

last_pts = None

mean_gmp_arr =np.array([])
vias = np.array([])
via_t = np.array([])
last_final_line, prior_line = None, None

class DraggablePoints:
    def __init__(self, points, on_update=None):
        """
        points: array Nx2 de posiciones iniciales.
        on_update: callback que recibe (pts_current:Nx2, changed_idxs:list or None) 
                en cada movimiento (si no hay cambios, changed_idxs será None).
        """
        self.points = np.array(points, dtype=float)
        self.last_points = self.points.copy()
        self.N = self.points.shape[0]
        self.on_update = on_update

        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.scatter = self.ax.scatter(self.points[:,0], self.points[:,1],
                                    s=100, c='red', picker=5)
        self.texts = []
        for i, (x, y) in enumerate(self.points):
            txt = self.ax.text(
                x, y, str(i+1), fontsize=12, ha='center', va='center',
                color='white',
                bbox=dict(boxstyle='circle', fc='blue', ec='none', alpha=0.6)
            )
            self.texts.append(txt)
        self.annot = self.ax.annotate(
            '', xy=(0,0), xytext=(10,10), textcoords='offset points',
            bbox=dict(boxstyle='round', fc='w'),
            arrowprops=dict(arrowstyle='->')
        )
        self.annot.set_visible(False)

        self.press = None
        self.ind = None
        c = self.fig.canvas
        c.mpl_connect('button_press_event',   self.on_press)
        c.mpl_connect('motion_notify_event',  self.on_motion)
        c.mpl_connect('button_release_event', self.on_release)

        self.ax.set_title('Move the vias by dragging them')
        self.ax.axis('equal')

    def on_press(self, event):
        if event.inaxes != self.ax: 
            return
        contains, info = self.scatter.contains(event)
        if not contains: 
            return
        self.ind = info['ind'][0]
        x0, y0 = self.points[self.ind]
        self.press = (x0, y0, event.xdata, event.ydata)
        self.annot.xy = (x0, y0)
        self.annot.set_text(f'Via-point {self.ind+1}')
        self.annot.set_visible(True)
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if self.press is None or self.ind is None or event.inaxes != self.ax:
            return
        x0, y0, xpress, ypress = self.press
        dx, dy = event.xdata - xpress, event.ydata - ypress
        new_pos = np.array([x0+dx, y0+dy])
        self.points[self.ind] = new_pos
        offsets = self.scatter.get_offsets()
        offsets[self.ind] = new_pos
        self.scatter.set_offsets(offsets)
        self.texts[self.ind].set_position(new_pos)
        self.annot.xy = new_pos
        self.fig.canvas.draw_idle()
        self._check_and_callback()

    def on_release(self, event):
        if self.press is None or self.ind is None:
            return
        self.press = None
        self.ind   = None
        self.annot.set_visible(False)
        self.fig.canvas.draw_idle()
        self._check_and_callback()

    def _check_and_callback(self):
        # detecta cambios comparando con last_points
        diffs = np.any(self.points != self.last_points, axis=1)
        changed_idxs = np.where(diffs)[0].tolist()
        # si no hay cambios, pasamos None
        callback_idxs = changed_idxs if changed_idxs else None
        # invocamos siempre el callback
        if self.on_update:
            self.on_update(self.points.copy(), callback_idxs)
        # actualizamos last_points
        self.last_points[:] = self.points

    def get_positions(self):
        return self.points.copy()

    def show(self):
        plt.show()

# Ejemplo de callback
def print_update(pts, changed):
    global last_changed, last_pts
    last_changed = changed      # lista de índices o None
    last_pts     = pts.copy()   # array Nx2
    
    #print('Posiciones actuales:\n', pts)
    #print('Índices movidos esta iteración:', 'None' if changed is None else [i+1 for i in changed])
    #print('-'*40)
    
    if last_changed is not None:
        pathwise_update_fn() 
    else:
        print("No hubo cambios en las posiciones.")
        

def pathwise_update_fn():
    global mean_gmp_arr, vias, via_t, last_changed, last_pts, last_final_line, prior_line
    BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential)
    base_kernels = [cls(lengthscales=1.0) for cls in BaseClasses]
    kernel = kernels.SeparateIndependent(base_kernels)

    F_prior_full = tf.expand_dims(
        tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()),
        axis=0
    )

    via_t = np.array([
        via_t[0],   
        via_t[-1]/3,
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

    # Dibujar nueva línea en el eje correcto (plt1)
    last_final_line, = plt.plot(mean_combo[:,0], mean_combo[:,1], '-.',color='red', linewidth=3, label='Final')
    prior_line,= plt.plot(mean_gmp_arr[0], mean_gmp_arr[1], '--', color='blue', linewidth=2, label='GMP Prior')
    
    plt.legend(fontsize=font_size)
    plt.xlabel('x [mm]', fontsize=font_size)
    plt.ylabel('y [mm]', fontsize=font_size)
    plt.axis('equal')
    #plt.figure.canvas.draw()
    #plt.show()


if __name__ == '__main__':
    #* Generation of GP prior
    
    np.random.seed(30)
    tf.random.set_seed(1)
    font_size = 18

    # Load dataset
    data = lasa.DataSet.GShape
    dt = data.dt
    demos = data.demos
    gap = 20
    demostraciones = 1

    # Prepare training data X, Y
    for i in range(demostraciones):
        demo = demos[i]
        pos = demo.pos[:, 0::gap]
        t = demo.t[:, 0::gap]
        X_ = t.T
        Y_ = pos.T
        if i == 0:
            X = X_
            Y = Y_
        else:
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))

    # Via/target points
    target_t = np.mean([demos[i].t[:,0::gap][0,-1] 
                        for i in range(demostraciones)])
    target_pos = np.mean([demos[i].pos[:,0::gap][:,-1] 
                        for i in range(demostraciones)], axis=0)
    via_t = np.array([
        np.mean([demos[i].t[:,0::gap][0,0] 
                for i in range(demostraciones)]),
        target_t
    ]).reshape(-1,1)
    vias = np.vstack((
        np.mean([demos[i].pos[:,0::gap][:,0] 
                for i in range(demostraciones)], axis=0),
        target_pos
    ))

    # Instantiate ProGpMp
    gp_mp = ProGpMp(X, Y, via_t, vias, dim=2, demos=demostraciones,
                    size=demos[0].pos[:, 0::gap].T.shape[0],
                    observation_noise=1.0)
    gp_mp.BlendedGpMp(gp_mp.ProGP)
    test_x = np.arange(0.0, target_t, dt)

    # GMP prediction
    t0 = time.time()
    mean_gmp, var_gmp = gp_mp.predict_BlendedPos(test_x.reshape(-1,1))
    t1 = time.time()
    print(f"Tiempo GPMP update: {t1 - t0:.6f} seconds")

    var_gmp[0] = np.where(var_gmp[0]<0, 0, var_gmp[0])
    var_gmp[1] = np.where(var_gmp[1]<0, 0, var_gmp[1])

    plt.figure(figsize=(16, 8), dpi=100)
    plt.subplots_adjust(left=0.05, right=0.99, wspace=0.8, hspace=0.8, bottom=0.1, top=0.99)
    plt1 = plt.subplot2grid((8, 16), (0, 0), rowspan=8, colspan=8)
    size = 80
    plt1.scatter(vias[:, 0], vias[:, 1], s=400, c='green', marker='x')
    plt1.scatter(Y[:, 0], Y[:, 1], s=30, c='green', marker='o', alpha=0.3)
    #plt1.plot(gp_mp1_predict_y_dim0, gp_mp1_predict_y_dim1, ls='-', c='blue', linewidth=2, label='$p_{gpmp1}$')

    plt1.plot(mean_gmp[0], mean_gmp[1], c='black', linewidth=4, label='$GMP$')
    #plt1.fill_between(mean_gmp[0], mean_gmp[1] - 5 * np.sqrt(abs(var_gmp[0])), mean_gmp[1] + 5 * np.sqrt(abs(var_gmp[1])), color='blue', alpha=0.5)
    #plt1.fill_between(X[:,0], Y[:,0] - 5 * np.sqrt(abs(X[:,0])), Y[:,0] + 5 * np.sqrt(abs(X[:,0])), color='red', alpha=0.5)

    plt1.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
    plt1.tick_params(labelsize=font_size)
    plt1.set_xlabel('$x$/mm\n(a)', fontsize=font_size)
    plt1.set_ylabel('$y$/mm', fontsize=font_size)

    """ print("Var Blendes:",var_gmp[0])
    print("Var Blendes2:",var_gmp[1]) """
    plt2 = plt.subplot2grid((8, 16), (0, 9), rowspan=3, colspan=8)
    plt2.plot(test_x, mean_gmp[0], c='red', linewidth=3, label='$x_{GMP}$')
    plt2.fill_between(test_x, mean_gmp[0] - 5 * np.sqrt(var_gmp[0]), mean_gmp[0] + 5 * np.sqrt(var_gmp[0]), color='red', alpha=0.3)
    plt2.scatter(via_t[:, 0], vias[:, 0], s=400, c='red', marker='x')
    plt2.scatter(X[:, 0], Y[:, 0], s=15, c='red', marker='o', alpha=0.3)
    plt2.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
    plt2.tick_params(labelsize=font_size)
    plt2.set_xlabel('$time$(s)\n(b)', fontsize=font_size)
    plt2.set_ylabel('$x$/mm', fontsize=font_size)

    plt3 = plt.subplot2grid((8, 16), (4, 9), rowspan=3, colspan=8)
    plt3.plot(test_x, mean_gmp[1], c='blue', linewidth=3, label='$y_{GMP}$')
    plt3.fill_between(test_x, mean_gmp[1] - 5 * np.sqrt(var_gmp[1]), mean_gmp[1] + 5 * np.sqrt(var_gmp[1]), color='blue', alpha=0.3)
    plt3.scatter(via_t[:, 0], vias[:, 1], s=400, c='blue', marker='x')
    plt3.scatter(X[:, 0], Y[:, 1], s=15, c='blue', marker='o', alpha=0.3)
    plt3.legend(loc='upper left', frameon=False, handlelength=1, ncol=3, columnspacing=1)
    plt3.tick_params(labelsize=font_size)
    plt3.set_xlabel('$time$(s)\n(c)', fontsize=font_size)
    plt3.set_ylabel('$y$/mm', fontsize=font_size)
    plt.show()
    old_mean = []
    old_mean = mean_gmp.copy()

    
    """ vias = np.array([
        [10, 20],   # vía 1
        [0, 0],     # vía 2
        [-20, -10]  # vía 3
    ]) """
    vias = np.array([
        vias[0],   # vía 1
        vias[-1]/3,
        vias[-1]/2,# vía 2
        vias[-1]  # vía 3
    ])
    
    
    mean_gmp_arr = np.array(mean_gmp, dtype=float)

    
    dp = DraggablePoints(vias, on_update=print_update)
    dp.show()

    # Como callback ya ha estado actualizando last_changed y last_pts,
    # a continuación puedes usarlos directamente en tu condición:
    print("Último last_changed:", last_changed)
    print("Último last_pts:\n", last_pts)
