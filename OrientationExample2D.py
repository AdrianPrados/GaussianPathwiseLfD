import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# GPflowSampling imports
from gpflow import kernels
from gpflow.config import default_float
from gpflow_sampling.sampling import updates

# Original imports
import pyLasaDataset as lasa
from ProGP import ProGpMp
import time

# Seed and visualization settings
np.random.seed(30)
tf.random.set_seed(1)
font_size = 18


# Load dataset
data = lasa.DataSet.Snake

dt = data.dt
demos = data.demos
gap = 40
demostraciones = 1

# Prepare training data X, Y (positions only)
for i in range(demostraciones):
    demo = demos[i]
    pos = demo.pos[:, 0::gap]
    t = demo.t[:, 0::gap]
    X_ = t.T
    Y_ = pos.T
    if i == 0:
        X, Y = X_, Y_
    else:
        X = np.vstack((X, X_)); Y = np.vstack((Y, Y_))

# Via/target points
target_t = np.mean([demos[i].t[:,0::gap][0,-1] for i in range(demostraciones)])
target_pos = np.mean([demos[i].pos[:,0::gap][:,-1] for i in range(demostraciones)], axis=0)
via_t = np.array([
    np.mean([demos[i].t[:,0::gap][0,0] for i in range(demostraciones)]),
    target_t
]).reshape(-1,1)
via_pos = np.vstack((
    np.mean([demos[i].pos[:,0::gap][:,0] for i in range(demostraciones)], axis=0),
    target_pos
))

# Instantiate and predict GMP (2D)
gp_mp = ProGpMp(X, Y, via_t, via_pos, dim=2, demos=demostraciones,
                size=Y.shape[0], observation_noise=1.0)
gp_mp.BlendedGpMp(gp_mp.ProGP)
test_x = np.arange(0.0, target_t, dt)
mean_list, var_list = gp_mp.predict_BlendedPos(test_x.reshape(-1,1))
mean_gmp = np.vstack(mean_list)
var_gmp = np.vstack(var_list)
for d in range(2): var_gmp[d] = np.clip(var_gmp[d], 0, None)


N = mean_gmp.shape[1]
#theta_prior0 = planar_slerp(init_orientation, final_orientation, N)
diffs = np.diff(mean_gmp.T, axis=0)
theta_prior = np.arctan2(diffs[:,0], diffs[:,1])
theta_prior = np.concatenate((theta_prior, [theta_prior[-1]]))


#* Pathwise Conditioning including orientation

F_prior = np.vstack((mean_gmp, theta_prior))  # (3,N)
F_prior_full = tf.expand_dims(tf.convert_to_tensor(F_prior.T, dtype=default_float()), axis=0)

# Extended via observations: define observation points and orientations
via_t_ext = np.array([
    via_t[0],
    #via_t[0] + (via_t[-1]-via_t[0]) / 2,
    via_t[-1]
]).reshape(-1,1)
""" via_pos_ext = np.array([
    via_pos[0],  # posición inicial
    [-20,-10],
    via_pos[-1]  # posición final
]) """
via_pos_ext = np.array([
    [25,27],  # posición inicial
    [4,-2]  # posición final
])
via_ori_ext = np.array([
    theta_prior[0],   # orient. inicial
    #theta_prior[100],    
    theta_prior[-1]   # orient. final
])
Y_obs_np = np.hstack((via_pos_ext, via_ori_ext.reshape(-1,1)))  # (3,3)((via_pos_ext, via_ori_ext.reshape(-1,1)))  # (3,3)

X_obs = tf.convert_to_tensor(via_t_ext, dtype=default_float())            # (N_obs,1)
Y_obs = tf.expand_dims(tf.convert_to_tensor(Y_obs_np, dtype=default_float()), axis=0)  # (1,N_obs,3)

# Extract F_obs from prior
obs_idx = [int(np.argmin(np.abs(test_x - t))) for t in via_t_ext[:,0]]
F_obs = tf.expand_dims(
    tf.convert_to_tensor(F_prior.T[obs_idx], dtype=default_float()), axis=0
)

# Kernel for 3 dims and update

kerns = [kernels.SquaredExponential(lengthscales=1.0) for _ in range(3)]
kernel = kernels.SeparateIndependent(kerns)

dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)


# Apply update over grid

dF_full = dF_fn(tf.convert_to_tensor(test_x.reshape(-1,1), dtype=default_float()))
F_cond = F_prior_full + dF_full
t0 = time.time()
mean_cond = tf.squeeze(F_cond, axis=0).numpy().T  # (3,N)
t1 = time.time()
print(f"Tiempo pathwise update: {t1 - t0:.6f} segundos")

# --- Visualization ---
# 1) Plot 2D trajectory with reference frames at via points
plt.figure(figsize=(6,6))
plt.plot(mean_gmp[0], mean_gmp[1], '--', label='GMP Prior')
# Plot conditioned trajectory
plt.plot(mean_cond[0], mean_cond[1], '-', label='Conditioned')
# Draw orientation frames at via points
# For initial and final via: draw U-shape box; for mid: arrow
arrow_length = np.linalg.norm(via_pos_ext[1] - via_pos_ext[0]) * 0.1  # scale arrow
# U-shape parameters
w, h = arrow_length, arrow_length*1.2  # width and height of U
for idx, ((x, y), theta) in enumerate(zip(via_pos_ext, via_ori_ext)):
    if idx == 1000:
        # middle: simple arrow
        """ dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=arrow_length*0.3, head_length=arrow_length*0.3, fc='red', ec='red') """
        print("Hey")
    else:
        # initial or final: draw U-shape
        local = np.array([
            [-w/2, h], [-w/2, 0], [-w/4, 0], [-w/4, h/2], [w/4, h/2], [w/4, 0], [w/2, 0], [w/2, h]
        ])
        if idx == 0:
            R = np.array([[np.sin(-theta), -np.cos(-theta)], [np.cos(-theta), np.sin(-theta)]])
        elif idx == 1:
            R = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
            
        pts = (R @ local.T).T + np.array([x, y])
        plt.fill(pts[:,0], pts[:,1], edgecolor='blue', facecolor='none', linewidth=2)
        # Also add arrow on U
        """ dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=arrow_length*0.3, head_length=arrow_length*0.3, fc='red', ec='red') """
        # Add orientation labels
        label = "Start" if idx == 0 else "End"
        offset = 0.02
        plt.text(x + offset, y + offset, label, fontsize=12, color='black')

""" plot_every = 5
for i, (xp, yp, thp) in enumerate(zip(mean_cond[0], mean_cond[1], mean_cond[2])):
    dxp = arrow_length * np.cos(thp)
    dyp = arrow_length * np.sin(thp)
    if i % plot_every == 0:
        plt.arrow(xp, yp, dxp, dyp,
                head_width=arrow_length*0.1, head_length=arrow_length*0.1,
                fc='gray', ec='gray', alpha=0.5) """

plt.scatter(via_pos_ext[:,0], via_pos_ext[:,1], c='red', marker='x', s=100, label='Via Points')
plt.axis('equal')
plt.legend()
plt.title('Conditioned Trajectory with Orientation Frames')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.show()

# 2) Plot each dimension (x, y) with Prior, Update and Corrected
# Prepare data for 1D plots
# time values
t_vals = test_x.squeeze()
# prior and conditioned arrays for positions only
f_prior = mean_gmp.T  # shape (N,2)
f_update = np.squeeze(dF_full.numpy()[0,:,:2])  # shape (N,2)
f_final = mean_cond[:2,:].T  # shape (N,2)
# via times and via positions
via_times = via_t_ext.squeeze()
vias_np = via_pos_ext  # shape (3,2) but only start and end relevant? we'll mark ext vias

# Create subplots: Prior, Update, Corrected
titles = ["Prior GMP", "Pathwise Update", "Corrected Path"]
colors = ['red', 'blue']
labels = ['x', 'y']
fig, axes = plt.subplots(figsize=(16, 4), ncols=3, sharey=True)

for dim in range(2):
    # Prior
    axes[0].plot(t_vals, f_prior[:, dim], label=f'{labels[dim]} prior', color=colors[dim])
    axes[0].scatter(via_times, vias_np[:, dim], color=colors[dim], marker='x', s=80)

    # Update (dF applies to pos dims)
    axes[1].plot(t_vals, f_update[:, dim], label=f'{labels[dim]} update', color=colors[dim])
    axes[1].scatter(via_times, np.zeros_like(via_times), color=colors[dim], marker='x', s=80)

    # Corrected
    axes[2].plot(t_vals, f_final[:, dim], label=f'{labels[dim]} corrected', color=colors[dim])
    axes[2].scatter(via_times, vias_np[:, dim], color=colors[dim], marker='x', s=80)

# Aesthetics
for i, ax in enumerate(axes):
    ax.axhline(0, linestyle='--', linewidth=0.5, color='gray')
    ax.set_xlabel('Time [s]')
    ax.set_title(titles[i])
    ax.grid(True)
    ax.legend()
axes[0].set_ylabel('Position [mm]')
plt.tight_layout()
plt.show()

# 3) Plot de la orientación (SLERP + Pathwise)
# Datos para orientación
f_prior_theta = theta_prior                  # (N,)
f_update_theta = np.squeeze(dF_full.numpy()[0,:,2])  # actualización
f_final_theta = mean_cond[2,:]               # corregido
via_ori_times = via_t_ext.squeeze()          # tiempos vía
via_ori_vals = via_ori_ext                   # orientaciones vía

# Subplots: Prior, Update, Corrected para theta
fig, axes = plt.subplots(1,3, figsize=(16,4), sharey=True)
titles_theta = ["Theta Prior", "Theta Update", "Theta Corrected"]
for i, ax in enumerate(axes):
    if i == 0:
        ax.plot(test_x, f_prior_theta, label='theta prior', color='purple')
        ax.scatter(via_ori_times, via_ori_vals, color='purple', marker='x', s=80)
    elif i == 1:
        ax.plot(test_x, f_update_theta, label='theta update', color='purple')
        ax.scatter(via_ori_times, np.zeros_like(via_ori_times), color='purple', marker='x', s=80)
    else:
        ax.plot(test_x, f_final_theta, label='theta corrected', color='purple')
        ax.scatter(via_ori_times, via_ori_vals, color='purple', marker='x', s=80)
    ax.axhline(0, linestyle='--', linewidth=0.5, color='gray')
    ax.set_xlabel('Time [s]')
    ax.set_title(titles_theta[i])
    ax.legend(); ax.grid(True)
axes[0].set_ylabel('Orientation (rad)')
plt.tight_layout()
plt.show()
