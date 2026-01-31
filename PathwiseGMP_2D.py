import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# GPflowSampling imports
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import priors, updates, decoupled
from gpflow_sampling.sampling.core import AbstractSampler

# Original imports
import pyLasaDataset as lasa
from ProGP import ProGpMp
import time

# Seed and visualization settings
np.random.seed(30)
tf.random.set_seed(1)
font_size = 18

# Load dataset
data = lasa.DataSet.heee
dt = data.dt
demos = data.demos
gap = 40
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
target_t = np.mean([demos[i].t[:,0::gap][0,-1] for i in range(demostraciones)])
target_pos = np.mean([demos[i].pos[:,0::gap][:,-1] for i in range(demostraciones)], axis=0)
via_t = np.array([np.mean([demos[i].t[:,0::gap][0,0] for i in range(demostraciones)]),
                target_t]).reshape(-1,1)
vias = np.vstack((np.mean([demos[i].pos[:,0::gap][:,0] for i in range(demostraciones)], axis=0),
                target_pos))

# Instantiate ProGpMp
gp_mp = ProGpMp(X, Y, via_t, vias, dim=2, demos=demostraciones,
                size=demos[0].pos[:, 0::gap].T.shape[0], observation_noise=1.0)
gp_mp.BlendedGpMp(gp_mp.ProGP)
test_x = np.arange(0.0, target_t, dt)

# GMP prediction
t0=time.time()
mean_gmp, var_gmp = gp_mp.predict_BlendedPos(test_x.reshape(-1,1))
t1 = time.time()
print(f"Tiempo pathwise update: {t1 - t0:.6f} segundos")
#time.sleep(1000)

var_gmp[0]=np.where(var_gmp[0]<0,0,var_gmp[0])
var_gmp[1]=np.where(var_gmp[1]<0,0,var_gmp[1])
#time.sleep(1000)
#print("VAriables blended:",var_gmp)


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


#* Pathwise optimization
BaseClasses = (kernels.SquaredExponential, kernels.SquaredExponential)
base_kernels = [cls(lengthscales=1.0) for cls in BaseClasses]
kernel = kernels.SeparateIndependent(base_kernels)


mean_gmp_arr = np.vstack(mean_gmp)      # shape (2, NT)
#NT = mean_gmp_arr.shape[1]

# 1) Preparamos el “prior” fijo a partir de mean_gmp
#    F_prior_full: tensor [1, NT, 2]
F_prior_full = tf.expand_dims(
    tf.convert_to_tensor(mean_gmp_arr.T, dtype=default_float()),
    axis=0
)



# 2) Montamos las observaciones (via points)
#    X_obs: [1, N_obs, 1], Y_obs: [1, N_obs, 2]
via_t = np.array([
    via_t[0],   
    np.array([1.8766]),
    via_t[-1]    
])

X_obs = tf.convert_to_tensor(via_t, dtype=default_float())

vias = np.array([
    [-30.4, 15],   # nuevo valor para el primer punto (start)
    [-14,7.8],
    [8.6, -6.5]    # nuevo valor para el segundo punto (target)
])

Y_obs = tf.expand_dims(tf.convert_to_tensor(vias, dtype=default_float()), axis=0)

# 3) Extraemos del prior sus valores en esos puntos
obs_idx = [int(np.argmin(np.abs(test_x - t))) for t in via_t[:,0]]
F_obs = tf.expand_dims(
    tf.convert_to_tensor(mean_gmp_arr.T[obs_idx], dtype=default_float()),
    axis=0
)  # shape [1, N_obs, 2]


# 4) Creamos la función de actualización pathwise
t0 = time.time()
print("Kernel:", kernel)
print("X_obs:", X_obs.shape)  # should be [1, N_obs, 1]
print("Y_obs:", Y_obs.shape)  # should be [1, N_obs, 2]
print("F_obs:", F_obs.shape)  # should be [1, N_obs, 2]
dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)
print("Df_fn:", dF_fn)

# 5) Muestreamos la corrección en toda la rejilla
X_test = test_x.reshape(-1,1)
X_test = tf.convert_to_tensor(test_x.reshape(-1, 1), dtype=default_float())  # (763, 1)
#X_test = tf.expand_dims(X_test, axis=0)  # (1, 763, 1)
dF_full = dF_fn(X_test)
F_cond = F_prior_full + dF_full
t1 = time.time()
print(f"Tiempo pathwise update: {t1 - t0:.6f} segundos")

print("**********************************++")
print("X_obs:", X_obs.shape)     # should be [1, N_obs, 1]
print("Y_obs:", Y_obs.shape)     # should be [1, N_obs, 2]
print("F_obs:", F_obs.shape)     # should be [1, N_obs, 2]
print("X_test:", X_test.shape)   # should be [1, NT, 1]
print("Shape dF-full:",dF_full.shape)
print("Shape Prior Full:",F_prior_full.shape)
print("Shape Final:",F_cond.shape)
#time.sleep(10000)
print("**********************************++")

mean_corrected = F_cond.numpy()

#print(mean_corrected)
#time.sleep(1000)


# 7) Visualization
# Ensure mean_gp_arr and mean_combo are 2D arrays (NT,2)
# If they have extra batch dims, squeeze them
mean_gp_arr = np.squeeze(mean_gmp_arr)
mean_combo = np.squeeze(mean_corrected)
print("mean_df:", dF_full)


plt.figure(figsize=(16,8), dpi=100)
# Plot original GMP prior
plt.plot(mean_gmp_arr[0], mean_gmp_arr[1], '--', linewidth=2, label='GMP Prior')
# Plot pathwise corrected
""" plt.plot(mean_df[0], mean_df[1], '-', linewidth=2, label='Pathwise Corrected') """
# Plot final blended
plt.plot(mean_combo[:,0], mean_combo[:,1], '-.', linewidth=3, label='Final')
# Vías/target points
plt.scatter(vias[:,0], vias[:,1], c='red', marker='x', s=100, label='Vías/Target')
plt.legend(fontsize=font_size)
plt.xlabel('x [mm]', fontsize=font_size)
plt.ylabel('y [mm]', fontsize=font_size)
plt.axis('equal')
plt.show()

#* Plot en 1D

# Subplots prior / update / corrected
fig, axes = plt.subplots(figsize=(16, 4), ncols=3, sharey=True)
titles = ["Prior GMP", "Pathwise Update", "Corrected Path"]
colors = ['red', 'blue']
labels = ['x', 'y']

# Prepara shapes correctos
t_vals = test_x.squeeze()                        # (763,)
f_prior = mean_gmp_arr.T                         # (763, 2)
f_update = np.squeeze(dF_full[0])                # (763, 2)
f_final = np.squeeze(mean_corrected)             # (763, 2)
via_times = via_t.squeeze()                      # (2,)
vias_np = vias                                   # (2, 2)

# Plot cada dimensión (x=0, y=1)
for dim in range(2):
    # Prior
    axes[0].plot(t_vals, f_prior[:, dim], label=f'{labels[dim]} prior', color=colors[dim])
    axes[0].scatter(via_times, vias_np[:, dim], color=colors[dim], marker='x', s=80)

    # Update
    axes[1].plot(t_vals, f_update[:, dim], label=f'{labels[dim]} update', color=colors[dim])
    axes[1].scatter(via_times, np.zeros_like(via_times), color=colors[dim], marker='x', s=80)

    # Corrected
    axes[2].plot(t_vals, f_final[:, dim], label=f'{labels[dim]} corrected', color=colors[dim])
    axes[2].scatter(via_times, vias_np[:, dim], color=colors[dim], marker='x', s=80)

# Estética
for i, ax in enumerate(axes):
    ax.axhline(0, linestyle='--', linewidth=0.5, color='gray')
    ax.set_xlabel('Time [s]')
    ax.set_title(titles[i])
    ax.grid(True)
    ax.legend()

axes[0].set_ylabel('Position [mm]')
plt.tight_layout()
plt.show()
