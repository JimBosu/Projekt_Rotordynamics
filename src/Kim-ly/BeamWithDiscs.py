# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:12:34 2025

@author: kimly
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# === USER INPUT FOR BEAM AND MATERIAL PROPERTIES ===
print("\n=== Define the beam and material properties ===")
L = float(input("Total beam length [m]: "))
E = float(input("Young's modulus [Pa]: "))
rho = float(input("Density [kg/m^3]: "))
D  = float(input("Diameter of beam [m]: "))
I = (np.pi * D**4) / 64     # Moment of inertia for circular cross-section [m^4]
A = (np.pi * D**2) / 4      # Cross-sectional area [m^2]


# === FEM discretization ===
n_elem = 10
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
disc1_pos = float(input("Disc 1 position [m]: "))   # [m] position along beam
disc1_diam = float(input("Disc 1 diameter [m]: "))   # [m] diameter of rigid disc
disc1_thick = float(input("Disc 1 thickness [m]: "))   # [m] axial thickness
disc1_density = float(input("Disc 1 density [kg/m^3]: "))   # [kg/m³] (same as beam material)

disc2_pos = float(input("Disc 2 position [m]: "))
disc2_diam = float(input("Disc 2 diameter [m]: "))
disc2_thick = float(input("Disc 2 thickness [m]: "))
disc2_density = float(input("Disc 2 density [kg/m^3]: "))


# === Add Discs ===
# Store all disc properties in a list of dictionaries
discs = [
    {'pos': disc1_pos, 'diameter': disc1_diam, 'thickness': disc1_thick, 'density': disc1_density},
    {'pos': disc2_pos, 'diameter': disc2_diam, 'thickness': disc2_thick, 'density': disc2_density}
]

# === ADD DISCS TO GLOBAL MASS MATRIX ===
for disc in discs:
    r = disc['diameter'] / 2
    m = disc['density'] * np.pi * r**2 * disc['thickness']  # Mass of the disc
    J = (1/4) * m * r**2 + (1/12) * m * disc['thickness']**2  # Rotational inertia about diameter axis (appropriate for lateral bending)

    # Map position to nearest node
    node = int(round(disc['pos'] / dx))
    node = np.clip(node, 0, n_nodes - 1)  # Ensure within bounds

    dof_v = 4 * node   # v (Y translation)
    dof_theta_z = dof_v + 1  # θz (rotation about Z)
    dof_w = dof_v + 2   # w (Z translation)
    dof_theta_y = dof_v + 3 # θy (rotation about Y)

    # Add disc mass to translational DOFs
    M_global[dof_v, dof_v] += m
    M_global[dof_w, dof_w] += m
    
    # Add disc rotational inertia to rotational DOFs
    M_global[dof_theta_y, dof_theta_y] += J
    M_global[dof_theta_z, dof_theta_z] += J


# === USER INPUT FOR FLEXIBLE BEARINGS ===
print("\n=== Define 2 Bearings ===")
bear1_pos = float(input("Bearing 1 position [m]: "))   # [m] position along beam
bear1_k = float(input("Bearing 1 translational stiffness [N/m]: ")) # [N/m] translational stiffness (flexibility of the bearing)
bear1_kr = float(input("Bearing 1 rotational stiffness [Nm/rad]: ")) # [Nm/rad] rotational stiffness of bearing (optional)

bear2_pos = float(input("Bearing 2 position [m]: "))
bear2_k = float(input("Bearing 2 translational stiffness [N/m]: "))
bear2_kr = float(input("Bearing 2 rotational stiffness [Nm/rad]: "))


# === Add Bearings ===
# Store bearing properties
bearings = [
    {'pos': bear1_pos, 'k': bear1_k, 'kr': bear1_kr},
    {'pos': bear2_pos, 'k': bear2_k, 'kr': bear2_kr}
]

# === ADD BEARING STIFFNESS TO GLOBAL STIFFNESS MATRIX ===
for b in bearings:
    node = int(round(b['pos'] / dx))
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

# === Solve eigenvalue problem ===
eigvals, eigvecs = eigh(K_reduced, M_reduced)
numerical_freqs = np.sqrt(eigvals) / (2 * np.pi)

# === Print natural frequencies ===
print("Numerical Natural frequencies (Hz):")
for i, f in enumerate(numerical_freqs[:6]):
    print(f"w_{i+1}: {f:.2f} Hz")

# === Plot mode shapes (u_y and u_z) ===
x = np.linspace(0, L, n_nodes)
n_modes = min(6, len(numerical_freqs))

fig = plt.figure(figsize=(10, 2.5 * n_modes)) 
for i in range(n_modes):
    full_mode = np.zeros(total_dof)
    full_mode[free_dofs] = eigvecs[:, i]
    
    v = full_mode[0::4]
    w = full_mode[2::4]
    
    ax = fig.add_subplot(n_modes, 1, i+1)
            # Plot mode shapes
    plt.plot(x, v, '-o', label='v (Y-dir)')
    plt.plot(x, w, '--', label='w (Z-dir)')
    
#================================================================================================================
    # === Mark disc positions ===
    for disc in discs:
        disc_node = int(round(disc['pos'] / dx))
        disc_x = disc_node * dx
        plt.axvline(x=disc_x, color='k', linestyle='--', label='Disc' if i == 0 else "")
    
    # === Mark bearing positions ===
    for b in bearings:
        bear_node = int(round(b['pos'] / dx))
        bear_x = bear_node * dx
        plt.axvline(x=bear_x, color='g', linestyle='-.', label='Bearing' if i == 0 else "")
#================================================================================================================
    plt.title(f"Mode {i+1}: {numerical_freqs[i]:.2f} Hz")
    plt.xlabel("Beam length (m)")
    plt.ylabel("Displacement")
    plt.legend(loc='upper right')
    plt.grid(True)

plt.tight_layout()
plt.show()