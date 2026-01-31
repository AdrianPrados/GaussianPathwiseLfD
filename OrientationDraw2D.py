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
from ProGP import ProGpMp

import time

# Seed and visualization settings
np.random.seed(30)
tf.random.set_seed(1)
font_size = 18

# --- SLERP helper for 2D angles ---
def planar_slerp(theta0, theta1, N):
    dtheta = (theta1 - theta0 + np.pi) % (2*np.pi) - np.pi
    u = np.linspace(0, 1, N)
    return theta0 + dtheta * u

# --- Interactive drawing to capture trajectory ---
print("Draw a trajectory with the mouse. Close the window when done.")
fig, ax = plt.subplots()
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_title('Draw trajectory: click and drag')
line, = ax.plot([], [], '-o')
xs, ys = [], []

def on_press(event):
    xs.clear(); ys.clear()
    line.set_data(xs, ys)
    fig.canvas.draw()

def on_move(event):
    if event.button == 1 and event.inaxes:
        xs.append(event.xdata)
        ys.append(event.ydata)
        line.set_data(xs, ys)
        fig.canvas.draw()

cid1 = fig.canvas.mpl_connect('button_press_event', on_press)
cid2 = fig.canvas.mpl_connect('motion_notify_event', on_move)
plt.show()
fig.canvas.mpl_disconnect(cid1)
fig.canvas.mpl_disconnect(cid2)

# Convert drawn data to arrays
N_pts = len(xs)
assert N_pts > 1, "Dibuja al menos dos puntos."
positions = np.vstack((xs, ys)).T
# Sampling reduction: downsample with gap
gap = 10
positions = positions[::gap]
# Update number of points
N_pts = positions.shape[0]

# Dummy time indices with constant dt
dt = 1.0
t0, tfinal = 0, dt*(N_pts-1)
t = np.arange(t0, tfinal+dt, dt)


# Define via points (initial, midpoint, final)
mid_idx = N_pts//2
#orientations = np.arctan2(np.diff(ys, prepend=ys[0]), np.diff(xs, prepend=xs[0]))
diffs = np.diff(positions, axis=0)
orientations = np.arctan2(diffs[:,1], diffs[:,0])
orientations = np.concatenate((orientations, [orientations[-1]]))
print(positions.shape)
print(orientations [0])
time.sleep(100000)
# Define via points (initial, midpoint, final)
mid_idx = N_pts//2

via_t_pc = np.array([[t0], [t[mid_idx]], [tfinal]]).reshape(-1,1)
via_pos_pc = np.vstack((positions[0], positions[mid_idx], positions[-1]))
via_ori_pc = np.array([orientations[0], orientations[mid_idx], orientations[-1]])

# GMP training data
X = t.reshape(-1,1)
Y = positions

dim = 2
# Instantiate and predict GMP
gp_mp = ProGpMp(X, Y, via_t_pc[:2], via_pos_pc[:2], dim=dim, demos=1,
                size=N_pts, observation_noise=1.0)
gp_mp.BlendedGpMp(gp_mp.ProGP)
mean_list, var_list = gp_mp.predict_BlendedPos(X)
mean_gmp = np.vstack(mean_list)
var_gmp = np.vstack(var_list)

# Orientation prior via SLERP between start and end orientations
#theta_prior = planar_slerp(orientations[0], orientations[-1], mean_gmp.shape[1])
theta_prior = orientations

#* Pathwise Conditioning including orientation
""" F_prior = np.vstack((mean_gmp, theta_prior))
F_prior_full = tf.expand_dims(tf.convert_to_tensor(F_prior.T, dtype=default_float()), axis=0)

# Observations for conditioning: full via (3 points)
X_obs = tf.convert_to_tensor(via_t_pc, dtype=default_float())
Y_obs_np = np.hstack((via_pos_pc, via_ori_pc.reshape(-1,1)))
Y_obs = tf.expand_dims(tf.convert_to_tensor(Y_obs_np, dtype=default_float()), axis=0)
obs_idx = [int(np.argmin(np.abs(t - tp))) for tp in via_t_pc[:,0]]
F_obs = tf.expand_dims(tf.convert_to_tensor(F_prior.T[obs_idx], dtype=default_float()), axis=0)


dim_kerns = [kernels.SquaredExponential(lengthscales=1.0) for _ in range(3)]
kernel = kernels.SeparateIndependent(dim_kerns)
dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)
dF_full = dF_fn(tf.convert_to_tensor(t.reshape(-1,1), dtype=default_float()))
F_cond = F_prior_full + dF_full
mean_cond = tf.squeeze(F_cond, axis=0).numpy().T """

F_prior = np.vstack((mean_gmp, theta_prior))  # (3,N)
F_prior_full = tf.expand_dims(tf.convert_to_tensor(F_prior.T, dtype=default_float()), axis=0)

via_t_ext = np.array([
    via_t_pc[0],
    via_t_pc[0] + (via_t_pc[-1]-via_t_pc[0]) / 2,
    via_t_pc[-1]
]).reshape(-1,1)
via_pos_ext = np.array([
    positions[0],  # posición inicial
    [-20,0],
    positions[-1]  # posición final
])
# Definir orientaciones deseadas en cada punto vía (rad)
# Puedes cambiar estos valores a mano o pedir input al usuario
via_ori_ext = np.array([
    orientations[0],   # orient. inicial
    orientations[0] + np.float64(0.78),    # orient. intermedia definida por el usuario
    orientations[-1]   # orient. final
])
Y_obs_np = np.hstack((via_pos_ext, via_ori_ext.reshape(-1,1)))  # (3,3)((via_pos_ext, via_ori_ext.reshape(-1,1)))  # (3,3)

X_obs = tf.convert_to_tensor(via_t_ext, dtype=default_float())            # (N_obs,1)
Y_obs = tf.expand_dims(tf.convert_to_tensor(Y_obs_np, dtype=default_float()), axis=0)  # (1,N_obs,3)

# Extract F_obs from prior
obs_idx = [int(np.argmin(np.abs(t - tp))) for tp in via_t_pc[:,0]]
F_obs = tf.expand_dims(
    tf.convert_to_tensor(F_prior.T[obs_idx], dtype=default_float()), axis=0
)

# Kernel for 3 dims and update
kerns = [kernels.SquaredExponential(lengthscales=1.0) for _ in range(3)]
kernel = kernels.SeparateIndependent(kerns)
dF_fn = updates.exact(kernel, X_obs, Y_obs, F_obs, diag=0.0)

# Apply update over grid
dF_full = dF_fn(tf.convert_to_tensor(t.reshape(-1,1), dtype=default_float()))
F_cond = F_prior_full + dF_full
mean_cond = tf.squeeze(F_cond, axis=0).numpy().T  # (3,N)

# --- Visualization ---
# 1) 2D trajectory with orientation frames
plt.figure(figsize=(6,6))
plt.plot(mean_gmp[0], mean_gmp[1], '--', label='GMP Prior')
plt.plot(mean_cond[0], mean_cond[1], '-', label='Conditioned')
# Draw frames
# Determine scale from drawing bounds
x_min, x_max = np.min(xs), np.max(xs)
print(xs, ys)
time.sleep(100000)
y_min, y_max = np.min(ys), np.max(ys)
traj_scale = max(x_max - x_min, y_max - y_min)
arrow_len = traj_scale * 0.05
w = traj_scale * 0.08
h = traj_scale * 0.06

for idx, ((x, y), theta) in enumerate(zip(via_pos_pc, via_ori_pc)):
    if idx == 1:
        # middle: simple arrow
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=arrow_len*0.3, head_length=arrow_len*0.3, fc='red', ec='red')
    else:
        # initial or final: draw U-shape
        local = np.array([
            [-w/2, h], [-w/2, 0], [-w/4, 0], [-w/4, h/2], [w/4, h/2], [w/4, 0], [w/2, 0], [w/2, h]
        ])
        if idx == 0:
            R = np.array([[np.sin(-theta), -np.cos(-theta)], [np.cos(-theta), np.sin(-theta)]])
        elif idx == 2:
            R = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
        pts = (R @ local.T).T + np.array([x, y])
        plt.fill(pts[:,0], pts[:,1], edgecolor='blue', facecolor='none', linewidth=2)
        # Also add arrow on U
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=arrow_len*0.3, head_length=arrow_len*0.3, fc='red', ec='red')
        # Add orientation labels
        label = "Inicio" if idx == 0 else "Final"
        offset = traj_scale * 0.02
        plt.text(x + offset, y + offset, label, fontsize=12, color='black')
plt.scatter(via_pos_pc[:,0], via_pos_pc[:,1], c='red', marker='x', s=100)
plt.axis('equal'); plt.legend(); plt.title('Trajectory with Frames')
plt.xlabel('x'); plt.ylabel('y'); plt.show()

# 2) Plot each dimension (x, y) with Prior, Update and Corrected
# Prepare data for 1D plots
# time values
t_vals = t.squeeze()
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
        ax.plot(t, f_prior_theta, label='theta prior', color='purple')
        ax.scatter(via_ori_times, via_ori_vals, color='purple', marker='x', s=80)
    elif i == 1:
        ax.plot(t, f_update_theta, label='theta update', color='purple')
        ax.scatter(via_ori_times, np.zeros_like(via_ori_times), color='purple', marker='x', s=80)
    else:
        ax.plot(t, f_final_theta, label='theta corrected', color='purple')
        ax.scatter(via_ori_times, via_ori_vals, color='purple', marker='x', s=80)
    ax.axhline(0, linestyle='--', linewidth=0.5, color='gray')
    ax.set_xlabel('Time [s]')
    ax.set_title(titles_theta[i])
    ax.legend(); ax.grid(True)
axes[0].set_ylabel('Orientation (rad)')
plt.tight_layout()
plt.show()
