import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
import pandas as pd
import time

# GPflowSampling imports
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import priors, updates, decoupled
from gpflow_sampling.sampling.core import AbstractSampler
from ProGP import ProGpMp

# quaternion to rotation matrix helper
def quat_to_rotmat(q):
    x, y, z, w = q
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])



#! ---- Code for real data from robots----
dt = 0.01
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
    qx_data = np.array(data['qx'][::gap])[:min_length]
    qy_data = np.array(data['qy'][::gap])[:min_length]
    qz_data = np.array(data['qz'][::gap])[:min_length]
    qw_data = np.array(data['qw'][::gap])[:min_length]
    
    data_dict[f"obstacle_{i}"] = np.array([x_data, y_data, z_data])
    quat_vals = np.array([qx_data, qy_data, qz_data, qw_data])
    
    if i == 1:
        Q = quat_vals.T  # shape (T, 4)
    else:
        Q = np.vstack((Q, quat_vals.T))
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
#!---- Code for RAIL datatset ----
""" def leer_archivos_mat(ruta_carpeta,number):
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
ruta_carpeta = '/home/adrian/Escritorio/ImitationLearning/GPFlow/FlowGMP/RAIL/REACHING'
number = 6
data = leer_archivos_mat(ruta_carpeta,number)

# Example of use: print the data from 1.mat
#print(data[number]['dataset'][0]['obj'][0].T)  #data[1]['dataset']['pos'][numero de demostraciones][0].T
#time.sleep(1000)
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
        Y = np.vstack((Y, vias)) """

np.random.seed(30)
font_size = 18
#* Selecting the target position and time and via points
#! Points for Real data (constructed by hand, you can use what ever you want)
target_t = t[0][-1]
target_position=np.array([values[0][0][-1], values[0][1][-1], values[0][2][-1]])

middle_t = t[0][min_length//2]
middle_position= np.array([x_data[min_length//2], y_data[min_length//2], z_data[min_length//2]])

via_point0_t = t[0][0]
via_point0_position= np.array([x_data[0], y_data[0], z_data[0],])

""" via_point1_t = t[0][106]
via_point1_position = np.array([values[2][0][106],values[2][1][106], values[2][2][106]]) """

#* Constructing the training set with the target and via points
#! Points for RAIL data
#size = demos[0].pos[:, 0::gap].T.shape[0]
""" target_t = sum(demos[i]['time'][0].T[:, 0::gap][0, -1] for i in range(demostraciones)) / demostraciones
target_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, -1]*10 for i in range(demostraciones)) / demostraciones
via_point0_t = sum(demos[i]['time'][0].T[:, 0::gap][0, 0] for i in range(demostraciones)) / demostraciones
via_point0_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, 0]*10 for i in range(demostraciones)) / demostraciones
via_point1_t = sum(demos[i]['time'][0].T[:, 0::gap][0, demos[i]['pos'][0].T[:, 0::gap].shape[1] * 2 // 10] for i in range(demostraciones)) / demostraciones
via_point1_position = sum(demos[i]['pos'][0].T[:, 0::gap][:, demos[i]['pos'][0].T[:, 0::gap].shape[1] * 2 // 10]*10 for i in range(demostraciones)) / demostraciones """
#Vias points

via_t = np.array([via_point0_t,middle_t, target_t]).reshape(-1, 1)
vias = np.array([via_point0_position, middle_position, target_position])

min_idx = min_length // 2
vias_quat = np.array([
    [qx_data[0],   qy_data[0],   qz_data[0],   qw_data[0]],
    [qx_data[min_idx], qy_data[min_idx], qz_data[min_idx], qw_data[min_idx]],
    [qx_data[-1],  qy_data[-1],  qz_data[-1],  qw_data[-1]],
])


#time.sleep(1000)
# predicting for dim0   --> size=demos[0]['pos'][0].T[:, 0::gap].T.shape[0]
observation_noise = 0.5
#gp_mp= ProGpMp(X, Y, via_t, vias,dim=3, demos=demostraciones,size = demos[0].pos[:, 0::gap].T.shape[0] , observation_noise=observation_noise) # For RAIL data use: demos[0].pos[:, 0::gap].T.shape[0]
gp_mp= ProGpMp(X, Y, via_t, vias,dim=3, demos=demostraciones,size = size, observation_noise=observation_noise) # For real data
gp_mp.BlendedGpMp(gp_mp.ProGP) #? If you use more than one GpMp is mandatory to use BlendedGpMp, input: list[]

gp_quat = ProGpMp(X, Q, via_t, vias_quat, dim=4, demos=demostraciones, size=min_length, observation_noise=0.01)
gp_quat.BlendedGpMp(gp_quat.ProGP)

test_x = np.arange(0.0, target_t, dt)
#test_x = np.arange(vias[0,0],vias[2,0],(1/pos0.size))


#alpha_list = (np.tanh((test_x - 0.5) * 5) + 1.0) / 2
#print(type(alpha_list))
alpha_list=np.ones(len(test_x))
alpha_list = np.vstack((alpha_list, alpha_list))
alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))

# alpha_list = np.vstack((np.ones(np.shape(test_x)[0]), np.ones(np.shape(test_x)[0])))
#* Predict position
mean_gmp, var_blended = gp_mp.predict_BlendedPos(test_x.reshape(-1, 1))

#* Predict quaternion
mean_quat, var_quat = gp_quat.predict_BlendedPos(test_x.reshape(-1, 1))

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
ax1.plot(mean_gmp[0], mean_gmp[1], mean_gmp[2], c='black', linewidth=5, label='$ProGpMp$')

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

#* Plot Orientation
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
quat_labels = ['qx', 'qy', 'qz', 'qw']
colors = ['red', 'blue', 'green', 'black']
for i, (label, col) in enumerate(zip(quat_labels, colors)):
    ax = axes[i]
    ax.plot(test_x, mean_quat[i], c=col, linewidth=3, label=f'${label}$')
    ax.fill_between(test_x,
                    mean_quat[i] - 5*np.sqrt(var_quat[i]),
                    mean_quat[i] + 5*np.sqrt(var_quat[i]),
                    color=col, alpha=0.3)
    ax.scatter(via_t[:,0], vias_quat[:,i], s=200, c=col, marker='x')
    ax.scatter(X[:,0], Q[:,i], s=10, c=col, marker='o', alpha=0.3)
    ax.set_ylabel(f'${label}$', fontsize=font_size)
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True)
axes[-1].set_xlabel('Time [s]', fontsize=font_size)
plt.suptitle('Quaternion Components 2D Visualization', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

old_mean = []
old_mean = mean_gmp.copy()

#! Pathwise optimization in 3D
BaseClasses = (kernels.SquaredExponential,)*3
base_kernels = [cls(lengthscales=1.0) for cls in BaseClasses]
kernel = kernels.SeparateIndependent(base_kernels)

BaseClassesQuat = (kernels.SquaredExponential,)*4
base_kernelsQuat = [cls(lengthscales=1.0) for cls in BaseClassesQuat]
kernelQuat = kernels.SeparateIndependent(base_kernelsQuat)


mean_gmp_arr = np.vstack(mean_gmp)      # shape (3, NT)

F_prior_full = tf.expand_dims(tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()),axis=0)

#* New vias point
vias_original = np.array([
    vias[0],   # nuevo valor para el primer punto (start)
    middle_position,
    vias[-1]    # nuevo valor para el segundo punto (target)
])
via_t = np.array([
    via_t[0],
    via_t[1],   
    via_t[-1]    
])

vias = np.array([
    vias[0]*1.25,   # nuevo valor para el primer punto (start)
    middle_position/(1.25),
    vias[-1]*1.25    # nuevo valor para el segundo punto (target)
])

via_t_quat = vias.copy()

vias_quat_original = np.array([
    [qx_data[0], qy_data[0], qz_data[0], qw_data[0]],
    [qx_data[min_idx], qy_data[min_idx], qz_data[min_idx], qw_data[min_idx]],
    [qx_data[-1], qy_data[-1], qz_data[-1], qw_data[-1]],
])
vias_quat = np.array([
    [qx_data[0]+0.45, qy_data[0]-0.10, qz_data[0]+0.65, qw_data[0]],
    [qx_data[min_idx]+0.17, qy_data[min_idx]-0.9, qz_data[min_idx]+0.74, qw_data[min_idx]],
    [qx_data[-1]+0.15, qy_data[-1]-0.35, qz_data[-1]-0.25, qw_data[-1]],
])


X_obs = tf.convert_to_tensor(via_t, dtype=default_float())
Y_obs = tf.expand_dims(tf.convert_to_tensor(vias, dtype=default_float()), axis=0)

# 3) Extraemos del prior sus valores en esos puntos
obs_idx = [int(np.argmin(np.abs(test_x - t))) for t in via_t[:,0]]
F_obs = tf.expand_dims(
    tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()),
    axis=0
)  # shape [1, N_obs, 2]


# 4) Creamos la función de actualización pathwise
t0 = time.time()
dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)
X_test = test_x.reshape(-1,1)
X_test = tf.convert_to_tensor(test_x.reshape(-1, 1), dtype=default_float())  # (763, 1)
#X_test = tf.expand_dims(X_test, axis=0)  # (1, 763, 1)
dF_full = dF_fn(X_test)
F_cond = F_prior_full + dF_full



mean_corrected = F_cond.numpy()
mean_gp_arr = np.squeeze(mean_gmp_arr)
mean_combo = np.squeeze(mean_corrected)

#* quaternion prior
mean_quat_arr = np.vstack(mean_quat)
F_prior_q = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T, dtype=default_float()), axis=0)
X_obs_q = tf.convert_to_tensor(via_t, dtype=default_float())
Y_obs_q = tf.expand_dims(tf.convert_to_tensor(vias_quat, dtype=default_float()), axis=0)
obs_idx_q = [int(np.argmin(np.abs(test_x - t))) for t in via_t[:,0]]
F_obs_q = tf.expand_dims(tf.convert_to_tensor(mean_quat_arr.T[obs_idx_q], dtype=default_float()), axis=0)
# pathwise update quat
dF_fn_q = updates.exact(kernelQuat, X_obs_q, Y_obs_q, F_obs_q, diag=0.0)
dF_full_q = dF_fn_q(X_test)
F_cond_q = F_prior_q + dF_full_q
mean_cond_q = np.squeeze(F_cond_q.numpy())
t1 = time.time()
print(f"Tiempo pathwise update: {t1 - t0:.6f} segundos")




fig = plt.figure(figsize=(10, 7))
ax3 = fig.add_subplot(111, projection='3d')

ax3.plot(
    mean_gmp_arr[0], mean_gmp_arr[1], mean_gmp_arr[2],
    '-',c='black', linewidth=2, label='GMP Prior'
)

# Plot final blend (línea punteada)
ax3.plot(
    mean_combo[:,0], mean_combo[:,1], mean_combo[:,2],
    '-.',c='orange', linewidth=3, label='Final')

# Scatter de los puntos vía
ax3.scatter(
    vias[:,0], vias[:,1], vias[:,2],
    c='red', marker='x', s=100, label='Vías/Target')

ax3.set_xlabel('x [mm]', fontsize=12)
ax3.set_ylabel('y [mm]', fontsize=12)
ax3.set_zlabel('z [mm]', fontsize=12)
ax3.legend(fontsize=12)
ax3.set_title('Trayectoria 3D GMP Pathwise', fontsize=14)



colors_vec = ['r','g','b']
for pt, qv in zip(vias, vias_quat):
    R = quat_to_rotmat(qv)
    for j, col in enumerate(colors_vec):
        # eje j coloreado: rojo, verde, azul para x,y,z
        ax3.quiver(pt[0], pt[1], pt[2], *(R[:,j]*1.0), color=col, length=1.0)

colors_vec = ['r','g','b']
for pt2, qv2 in zip(vias_original, vias_quat_original):
    R2 = quat_to_rotmat(qv2)
    for j, col in enumerate(colors_vec):
        # eje j coloreado: rojo, verde, azul para x,y,z
        ax3.quiver(pt2[0], pt2[1], pt2[2], *(R2[:,j]*1.0), color=col, length=1.0)

plt.tight_layout()
plt.show()

#* Plot en 1D

# Subplots prior / update / corrected
fig, axes = plt.subplots(figsize=(16, 4), ncols=3, sharey=True)
titles = ["Prior GMP", "Pathwise Update", "Corrected Path"]
labels = ['x', 'y', 'z']

# Prepara shapes correctos
t_vals = test_x.squeeze()                        
f_prior = mean_gmp_arr.T                         
f_update = np.squeeze(dF_full[0])                
f_final = np.squeeze(mean_corrected)             
via_times = via_t.squeeze()                      
vias_np = vias


for dim in range(3):
    # Prior
    axes[0].plot(t_vals, f_prior[:, dim], label=f'{labels[dim]} prior', color=colors[dim])
    axes[0].scatter(via_times, vias_np[:, dim], color=colors[dim], marker='x', s=80)

    # Update
    axes[1].plot(t_vals, f_update[:, dim], label=f'{labels[dim]} update', color=colors[dim])
    axes[1].scatter(via_times, np.zeros_like(via_times), color=colors[dim], marker='x', s=80)

    # Corrected
    axes[2].plot(t_vals, f_final[:, dim], label=f'{labels[dim]} corrected', color=colors[dim])
    axes[2].scatter(via_times, vias_np[:, dim], color=colors[dim], marker='x', s=80)
    

for i, ax in enumerate(axes):
    ax.axhline(0, linestyle='--', linewidth=0.5, color='gray')
    ax.set_xlabel('Time [s]')
    ax.set_title(titles[i])
    ax.grid(True)
    ax.legend()

axes[0].set_ylabel('Position [mm]')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(figsize=(16, 4), ncols=3, sharey=True)
titles = ["Prior GMP Orientation", "Pathwise Update Orientation", "Corrected Path Orientation"]
labels = ['qx', 'qy', 'qz', 'qw']


f_priorQuat = mean_quat_arr.T
f_updateQuat = np.squeeze(dF_full_q[0])
f_finalQuat = np.squeeze(mean_cond_q)

for dim_q in range(4):
    # Prior
    axes[0].plot(t_vals, f_priorQuat[:, dim_q], label=f'{labels[dim_q]} prior', color=colors[dim_q])
    axes[0].scatter(via_times, vias_quat[:, dim_q], color=colors[dim_q], marker='x', s=80)

    # Update
    axes[1].plot(t_vals, f_updateQuat[:, dim_q], label=f'{labels[dim_q]} update', color=colors[dim_q])
    axes[1].scatter(via_times, np.zeros_like(via_times), color=colors[dim_q], marker='x', s=80)

    # Corrected
    axes[2].plot(t_vals, f_finalQuat[:, dim_q], label=f'{labels[dim_q]} corrected', color=colors[dim_q])
    axes[2].scatter(via_times, vias_quat[:, dim_q], color=colors[dim_q], marker='x', s=80)

for i, ax in enumerate(axes):
    ax.axhline(0, linestyle='--', linewidth=0.5, color='gray')
    ax.set_xlabel('Time [s]')
    ax.set_title(titles[i])
    ax.grid(True)
    ax.legend()
axes[0].set_ylabel('Orientation [rads]')
plt.tight_layout()
plt.show()
