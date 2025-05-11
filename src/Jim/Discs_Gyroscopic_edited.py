# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:14:36 2025

@author: kimly, busoj
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# === USER INPUT FOR BEAM AND MATERIAL PROPERTIES ===
print("\n=== Define the beam and material properties ===")
L = 2.4384 # float(input("Total beam length [m]: "))
Dinner = 0.0125 # float(input("Inner diameter of beam [m]: "))
Douter  = 0.051 # float(input("Outer diameter of beam [m]: "))
rho = 7.8334E-6 # float(input("Density [kg/m^3]: "))
E = 2.0684E8 # float(input("Young's modulus [Pa]: "))
Omega = 1000/60*2*np.pi # float(input("Spin speed [rad/s]: "))

I = (np.pi / 64) * (Douter**4 - Dinner**4)  # Moment of inertia for hollow circular cross-section [m^4]
A = (np.pi / 4) * (Douter**2 - Dinner**2)   # Cross-sectional area for hollow cylinder [m^2]


# === FEM discretization ===
n_elem = 24
n_nodes = n_elem + 1
dof_per_node = 4                            # 4 DOF: u_y, theta_z, u_z, theta_y
total_dof = dof_per_node * n_nodes
dx = L / n_elem                             # Element length

# === Element matrices (8x8) ===
def beam_element_matrices_3D(E, I, rho, A, L):
    # 4x4 stiffness matrix for bending
    K_local = (E * I / L**3) * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2]
    ])
    
    # 4x4 consistent mass matrix
    M_local = (rho * A * L / 420) * np.array([
        [156, 22*L, 54, -13*L],
        [22*L, 4*L**2, 13*L, -3*L**2],
        [54, 13*L, 156, -22*L],
        [-13*L, -3*L**2, -22*L, 4*L**2]
    ])
    
    # 8x8 Element matrices in 3D
    K_e = np.zeros((8, 8))
    M_e = np.zeros((8, 8))
    
    # Fill 8x8 matrices (see notes)
    K_e[np.ix_([0,1,4,5],[0,1,4,5])] = K_local
    M_e[np.ix_([0,1,4,5],[0,1,4,5])] = M_local
    
    K_e[np.ix_([2,3,6,7],[2,3,6,7])] = K_local
    M_e[np.ix_([2,3,6,7],[2,3,6,7])] = M_local
    
    
    return K_e, M_e

# === Initialize global matrices ===
K_global = np.zeros((total_dof, total_dof))
M_global = np.zeros((total_dof, total_dof))
G_global = np.zeros((total_dof, total_dof))


# === Assembly ===
for e in range(n_elem):
    k_e, m_e = beam_element_matrices_3D(E, I, rho, A, dx)
    dof_map = [
        4*e, 4*e+1, 4*e+2, 4*e+3,
        4*e+4, 4*e+5, 4*e+6, 4*e+7
    ]
    
    for i in range(8):
        for j in range(8):
            K_global[dof_map[i], dof_map[j]] += k_e[i, j]
            M_global[dof_map[i], dof_map[j]] += m_e[i, j]
                          

#=============================================================================================================================
# === USER INPUT FOR 2 DISCS ===
print("\n=== Define 2 Rigid Discs ===")
disc1_pos = 10  # float(input("Disc 1 position [m]: "))   # [m] position along beam
disc1_diam = 0.1047 # float(input("Disc 1 diameter [m]: "))   # [m] diameter of rigid disc
disc1_thick = 0.0506 # float(input("Disc 1 thickness [m]: "))   # [m] axial thickness
disc1_density = 7.8334E-6 # float(input("Disc 1 density [kg/m^3]: "))   # [kg/m³] (same as beam material)

disc2_pos = 13 # float(input("Disc 2 position [m]: "))
disc2_diam = 0.1047 # float(input("Disc 2 diameter [m]: "))
disc2_thick = 0.0506 # float(input("Disc 2 thickness [m]: "))
disc2_density = 7.8334E-6 # float(input("Disc 2 density [kg/m^3]: "))

disc3_pos = 16 # float(input("Disc 2 position [m]: "))
disc3_diam = 0.1047 # float(input("Disc 2 diameter [m]: "))
disc3_thick = 0.0506 # float(input("Disc 2 thickness [m]: "))
disc3_density = 7.8334E-6 # float(input("Disc 2 density [kg/m^3]: "))


# === Add Discs ===
# Store all disc properties in a list of dictionaries
discs = [
    {'pos': disc1_pos, 'diameter': disc1_diam, 'thickness': disc1_thick, 'density': disc1_density},
    {'pos': disc2_pos, 'diameter': disc2_diam, 'thickness': disc2_thick, 'density': disc2_density},
    {'pos': disc3_pos, 'diameter': disc3_diam, 'thickness': disc3_thick, 'density': disc3_density}
]

# === ADD DISCS TO GLOBAL MASS MATRIX ===
for disc in discs:
    r = disc['diameter'] / 2
    m = disc['density'] * np.pi * r**2 * disc['thickness']  # Mass of the disc
    J = (1/4) * m * r**2 + (1/12) * m * disc['thickness']**2  # Rotational inertia about diameter axis (appropriate for lateral bending)
    I = 0.5 * m * r**2 # Polar moment of inertia for the disc (used in gyroscopic effects)

    # Map position to nearest node
    node = np.clip(disc['pos'], 0, n_nodes - 1)  # Ensure within bounds

    dof_uy = 4 * node   # v (Y translation)
    dof_theta_z = dof_uy + 1  # θz (rotation about Z)
    dof_uz= dof_uy + 2   # w (Z translation)
    dof_theta_y = dof_uy + 3 # θy (rotation about Y)

    # Add disc mass to translational DOFs
    M_global[dof_uy, dof_uy] += m
    M_global[dof_uz, dof_uz] += m
    
    # Add disc rotational inertia to rotational DOFs
    M_global[dof_theta_y, dof_theta_y] += J
    M_global[dof_theta_z, dof_theta_z] += J
    
    # Add gyroscopic matrix contributions
    G_local = np.array([
        [0,      0,      0, -I * Omega],
        [0,      0,  I * Omega, 0],
        [0, -I * Omega, 0,     0],
        [I * Omega, 0, 0,      0]
    ])

    # Map local [u_y, theta_z, u_z, theta_y] into global
    dof_indices = [dof_uy, dof_theta_z, dof_uz, dof_theta_y]

    for i in range(4):
        for j in range(4):
            G_global[dof_indices[i], dof_indices[j]] += G_local[i, j]


# === USER INPUT FOR FLEXIBLE BEARINGS ===
print("\n=== Define 2 Bearings ===")
bear1_pos = 4 # float(input("Bearing 1 position [m]: "))   # [m] position along beam
bear1_k = 1e6 # float(input("Bearing 1 translational stiffness [N/m]: ")) # [N/m] translational stiffness (flexibility of the bearing)
bear1_kr = 1e6 # float(input("Bearing 1 rotational stiffness [Nm/rad]: ")) # [Nm/rad] rotational stiffness of bearing (optional)

bear2_pos = 22 # float(input("Bearing 2 position [m]: "))
bear2_k = 1e6 # float(input("Bearing 2 translational stiffness [N/m]: "))
bear2_kr = 1e6 # float(input("Bearing 2 rotational stiffness [Nm/rad]: "))


# === Add Bearings ===
# Store bearing properties
bearings = [
    {'pos': bear1_pos, 'k': bear1_k, 'kr': bear1_kr},
    {'pos': bear2_pos, 'k': bear2_k, 'kr': bear2_kr}
]

# === ADD BEARING STIFFNESS TO GLOBAL STIFFNESS MATRIX ===

for b in bearings:
    node = int(b['pos'])
    node = np.clip(node, 0, n_nodes - 1)

    dof_v = 4 * node   # v (Y direction)
    dof_theta_z = dof_v + 1   # θz (about Z)
    dof_w = dof_v + 2    # w (Z direction)
    dof_theta_y = dof_v + 3   # θy (about Y)
    
    # Add translational stiffness
    K_global[dof_v, dof_v] += b['k']
    K_global[dof_w, dof_w] += b['k']
    
    # Add rotational stiffness
    K_global[dof_theta_y, dof_theta_y] += b['kr']
    K_global[dof_theta_z, dof_theta_z] += b['kr']
    

    
#=============================================================================================================================  
   
# === Boundary conditions (simply supported at both ends in Y and Z) ===
# Fix displacements u_y, u_z at node 0 and node n_nodes-1
constrained_dofs = [
    0, 2,                      # u_y0, u_z0
    dof_per_node*(n_nodes-1),  # u_y at last node
    dof_per_node*(n_nodes-1)+2 # u_z at last node
]
free_dofs = np.setdiff1d(np.arange(total_dof), constrained_dofs)

# === Reduce matrices ===
K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
M_reduced = M_global[np.ix_(free_dofs, free_dofs)]
G_reduced = G_global[np.ix_(free_dofs, free_dofs)]
I = np.eye(M_reduced.shape[0])                  # Identity matrix

# Build A and B matrices
A_mat = np.block([
    [G_reduced, M_reduced],
    [I, np.zeros_like(M_reduced)]
])

B_mat = np.block([
    [K_reduced, np.zeros_like(M_reduced)],
    [np.zeros_like(M_reduced), -I]
])

# Solve generalized eigenvalue problem: A x_dot + B x = 0
eigvals, eigvecs = eig(-B_mat, A_mat)  # Notice negative sign to match equation

# Extract natural frequencies from eigenvalues
omega = np.abs(np.imag(eigvals))  # rad/s
numerical_freqs = omega / (2 * np.pi)       # Hz

# Ordering natural frequencies from lowest to highest
sorted_indices = np.argsort(numerical_freqs)
eigvecs = eigvecs[:, sorted_indices]
numerical_freqs = numerical_freqs[sorted_indices]


# ====================================================== Print natural frequencies ======================================================
print("Numerical Natural frequencies (Hz):")
for i, f in enumerate(numerical_freqs[:12]):
    print(f"w_{i+1}: {f:.2f} Hz")

# === Plot mode shapes (u_y and u_z) ===
# ====================================================== PLot 3D ======================================================

# Define number of modeshapes plotted
n_modes = min(6, len(numerical_freqs))
x = np.linspace(0, L, n_nodes)

# creating a 3D Disc
def create_disc(x_center, y_center, z_center, radius, num_points=30):
    theta = np.linspace(0, 2 * np.pi, num_points)
    y = y_center + radius * np.cos(theta)
    z = z_center + radius * np.sin(theta)
    x = np.full_like(y, x_center)
    return list(zip(x, y, z))

# 3D-Plot for every modeshape
for i in range(n_modes):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    full_mode = np.zeros(total_dof)
    eigvec_displacement = eigvecs[M_reduced.shape[0]:, i]
    full_mode[free_dofs] = np.abs(eigvec_displacement)

    v = full_mode[0::4]
    w = full_mode[2::4]

    y_def = v
    z_def = w

    # Axis not deformed
    ax.plot(x, np.zeros_like(x), np.zeros_like(x), 'k--', linewidth=1, label='Axis')

    # Axis defomrmed
    ax.plot(x, y_def, z_def, 'y.-', label=f'Mode {i+1}')

    # Discs
    for disc in discs:
        node = int(disc['pos'])
        disc_points = create_disc(x[node], y_def[node], z_def[node], radius=0.05)
        disc_poly = Poly3DCollection([disc_points], color='skyblue', alpha=0.5)
        ax.add_collection3d(disc_poly)

    # Bearings
    for b in bearings:
        node = int(b['pos'])
        ax.scatter(x[node], y_def[node], z_def[node], c='r', s=60, label='Bearing' if b == bearings[0] else "")

    # Nodes
    ax.scatter(x, y_def, z_def, c='yellow', edgecolors='k', s=30, label='Node')
    for idx in range(n_nodes):
        ax.text(x[idx], y_def[idx], z_def[idx], str(idx), fontsize=6)

    ax.set_xlim(0, L)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.2, 0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Mode Shape {i+1} ({numerical_freqs[i]:.2f} Hz)")
    ax.view_init(elev=20, azim=135)
    ax.legend()

    plt.tight_layout()
    plt.show()



# ====================================================== PLot 2D ======================================================
x = np.linspace(0, L, n_nodes)
n_modes = min(6, len(numerical_freqs))

fig = plt.figure(figsize=(10, 2.5 * n_modes)) 
for i in range(n_modes):
    full_mode = np.zeros(total_dof)
    eigvec_displacement = eigvecs[M_reduced.shape[0]:, i]
    full_mode[free_dofs] = np.abs(eigvec_displacement)

    v = full_mode[0::4]
    w = full_mode[2::4]

    ax = fig.add_subplot(n_modes, 1, i+1)
    plt.plot(x, v, '-o', label='u_y')
    plt.plot(x, w, '--', label='u_z')

    # === Mark disc positions (now using node indices directly) ===
    for disc in discs:
        disc_node = int(disc['pos'])
        disc_x = x[disc_node]
        plt.axvline(x=disc_x, color='k', linestyle='--', label='Disc' if i == 0 else "")
        plt.plot(disc_x, 0, 'ko', markerfacecolor='blue')

    # === Mark bearing positions (also node indices) ===
    for b in bearings:
        bear_node = int(b['pos'])
        bear_x = x[bear_node]
        plt.axvline(x=bear_x, color='g', linestyle='-.', label='Bearing' if i == 0 else "")
        plt.plot(bear_x, 0, 'go', markerfacecolor='orange')

    plt.title(f"Mode {i+1}: {numerical_freqs[i]:.2f} Hz")
    plt.xlabel("Beam length (m)")
    plt.ylabel("Displacement")
    plt.legend(loc='upper right')
    plt.grid(True)

plt.tight_layout()
plt.show()
