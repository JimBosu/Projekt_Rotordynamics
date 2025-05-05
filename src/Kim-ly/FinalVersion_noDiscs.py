# -*- coding: utf-8 -*-
"""
Created on Sun May  4 14:47:49 2025

@author: kimly
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Beam and material properties
L = 1.0                     # Total beam length [m]
E = 210e9                   # Young's modulus [Pa]
rho = 7800                  # Density [kg/m^3]
D = 0.02                    # Diameter of beam [m]
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
    
    
    # # 8x8 block diagonal matrices (bending in Y and Z planes)
    # K_e = np.block([
    #     [K_local, np.zeros((4,4))],
    #     [np.zeros((4,4)), K_local]
    # ])
    
    # M_e = np.block([
    #     [M_local, np.zeros((4,4))],
    #     [np.zeros((4,4)), M_local]
    # ])
    
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

plt.figure(figsize=(10, 2.5 * n_modes)) 

for i in range(n_modes):
    full_mode = np.zeros(total_dof)
    full_mode[free_dofs] = eigvecs[:, i]
    
    w_y = full_mode[::dof_per_node]       # u_y DOFs
    w_z = full_mode[2::dof_per_node]      # u_z DOFs
    #w_y /= np.max(np.abs(w_y))            # normalize
    #w_z /= np.max(np.abs(w_z))
    
    plt.subplot(n_modes, 1, i+1)
    plt.plot(x, w_y, '-o', label='u_y')
    plt.plot(x, w_z, '-o', label='u_z')
    plt.title(f"w_{i+1} - {numerical_freqs[i]:.2f} Hz")
    plt.xlabel("x [m]")
    plt.ylabel("Displacement (normalized)")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
