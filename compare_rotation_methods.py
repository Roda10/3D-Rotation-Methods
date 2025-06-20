import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# === Triangle setup ===
vertices = np.array([
    [1.0, 0.0, 0.0],  # A
    [0.0, 1.0, 0.0],  # B
    [0.0, 0.0, 1.0]   # C
])
labels = ['A', 'B', 'C']

# === Rotation helpers ===
def rot_x(a): r = np.radians(a); return np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
def rot_y(a): r = np.radians(a); return np.array([[np.cos(r),0,np.sin(r)],[0,1,0],[-np.sin(r),0,np.cos(r)]])
def rot_z(a): r = np.radians(a); return np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])

def rodrigues_matrix(axis, theta_deg):
    theta = np.radians(theta_deg)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta)) * (K @ K)

def quaternion_matrix(axis, angle_deg):
    theta = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis * np.sin(theta / 2)
    w = np.cos(theta / 2)
    return np.array([
        [1-2*(y**2+z**2),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x**2+z**2),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x**2+y**2)]
    ])

# === Configuration ===
axis = np.array([1, 1, 1])
angle_max = 120
frames = 60
fig = plt.figure(figsize=(18,6))
axs = [fig.add_subplot(131+i, projection='3d') for i in range(3)]
titles = ["Euler Angles (X→Y→Z)", "Rodrigues Rotation", "Quaternion Rotation"]
colors = ['skyblue', 'orange', 'violet']
ghost_alpha = 0.15

for ax, title in zip(axs, titles):
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.set_zlim([-1.5,1.5])
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=30, azim=45)
    ax.set_title(title, fontsize=11)
    ax.quiver(0,0,0,1,0,0,color='red')
    ax.quiver(0,0,0,0,1,0,color='green')
    ax.quiver(0,0,0,0,0,1,color='blue')
    ax.plot([0,axis[0]],[0,axis[1]],[0,axis[2]],'--',color='black',linewidth=1)

# === Initial patches, labels, trails ===
patches, texts, trails = [], [], [[] for _ in range(3)]
for ax, color in zip(axs, colors):
    p = Poly3DCollection([vertices], alpha=0.7, facecolor=color, edgecolor='black')
    ax.add_collection3d(p)
    patches.append(p)
    texts.append([ax.text(*pt, labels[i], fontsize=9, color='black') for i, pt in enumerate(vertices)])

angle_texts = [ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=10) for ax in axs]

# === Update function ===
def update(frame):
    theta = (frame / (frames-1)) * angle_max

    # Euler
    Rx = rot_x(min(theta, 90))
    Ry = rot_y(min(max(theta-90, 0), 20))
    Rz = rot_z(min(max(theta-110, 0), 10))
    Reuler = Rz @ Ry @ Rx

    # Rodrigues & Quaternion
    Rrod = rodrigues_matrix(axis, theta)
    Rquat = quaternion_matrix(axis, theta)

    rotations = [Reuler, Rrod, Rquat]

    for i, (R, patch, text_list, trail, ax, a_text) in enumerate(zip(rotations, patches, texts, trails, axs, angle_texts)):
        new_verts = vertices @ R.T
        patch.set_verts([new_verts])

        # Update labels
        for j, txt in enumerate(text_list):
            txt.set_position((new_verts[j][0], new_verts[j][1]))
            txt.set_3d_properties(new_verts[j][2], zdir='z')

        # Store trail
        trail.append(new_verts.copy())
        if len(trail) > 1:
            ghost = Line3DCollection([[p[i], p[(i+1)%3]] for p in trail[-5:]], alpha=ghost_alpha, colors='gray', linewidths=1)
            ax.add_collection3d(ghost)

        # Angle text
        a_text.set_text(f"θ = {theta:.1f}°")

    return patches + sum(texts, []) + angle_texts

# === Animate and save ===
ani = FuncAnimation(fig, update, frames=frames, interval=70, blit=False)

ani.save("compare_rotation_methods.gif", writer=PillowWriter(fps=20))
ani.save("compare_rotation_methods.mp4", writer=FFMpegWriter(fps=20))

fig.text(0.5, 0.02, "Visualization by Rodeo Oswald", ha='center', fontsize=12, style='italic')
plt.tight_layout()
plt.show()
