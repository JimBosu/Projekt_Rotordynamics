# -*- coding: utf-8 -*-
"""
Created on Mon May  5 09:37:07 2025

@author: kimly
"""

"""
CORRECTION: Solving per plane (two generalized eigenvalue problem)
Two global matrices → one for Y-plane bending, one for Z-plane bending.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Beam and material properties
L = 1.0                     # Total beam length [m]
E = 210e9                   # Young's modulus [Pa]
rho = 7800                  # Density [kg/m^3]
D = 0.02                    # Diameter of beam [m]
I = (np.pi * D**4) / 64     # Moment of inertia [m^4]
A = (np.pi * D**2) / 4      # Cross-sectional area [m^2]

# === FEM discretization ===
n_elem = 10
n_nodes = n_elem + 1
dof_per_node = 2  # per plane: [u, theta]
total_dof = dof_per_node * n_nodes
dx = L / n_elem

# === Element stiffness and mass matrices (4x4 for each plane) ===
def beam_element_matrices(E, I, rho, A, L):
    K_local = (E * I / L**3) * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2]
    ])
    
    M_local = (rho * A * L / 420) * np.array([
        [156, 22*L, 54, -13*L],
        [22*L, 4*L**2, 13*L, -3*L**2],
        [54, 13*L, 156, -22*L],
        [-13*L, -3*L**2, -22*L, 4*L**2]
    ])
    
    return K_local, M_local

# === Initialize global matrices for Y and Z planes ===
K_global_y = np.zeros((total_dof, total_dof))
M_global_y = np.zeros((total_dof, total_dof))

K_global_z = np.zeros((total_dof, total_dof))
M_global_z = np.zeros((total_dof, total_dof))

# === Assembly ===
K_local, M_local = beam_element_matrices(E, I, rho, A, dx)

for e in range(n_elem):
    dof_map = [2*e, 2*e+1, 2*e+2, 2*e+3]
    
    for i in range(4):
        for j in range(4):
            K_global_y[dof_map[i], dof_map[j]] += K_local[i, j]
            M_global_y[dof_map[i], dof_map[j]] += M_local[i, j]
            K_global_z[dof_map[i], dof_map[j]] += K_local[i, j]
            M_global_z[dof_map[i], dof_map[j]] += M_local[i, j]

# === Boundary conditions (simply supported at both ends) ===
# Fix displacement DOF at node 0 and last node
constrained_dofs = [0, dof_per_node*(n_nodes-1)]

free_dofs = np.setdiff1d(np.arange(total_dof), constrained_dofs)

# === Reduce matrices ===
K_reduced_y = K_global_y[np.ix_(free_dofs, free_dofs)]
M_reduced_y = M_global_y[np.ix_(free_dofs, free_dofs)]

K_reduced_z = K_global_z[np.ix_(free_dofs, free_dofs)]
M_reduced_z = M_global_z[np.ix_(free_dofs, free_dofs)]

# === Solve eigenvalue problem for Y and Z planes ===
eigvals_y, eigvecs_y = eigh(K_reduced_y, M_reduced_y)
eigvals_z, eigvecs_z = eigh(K_reduced_z, M_reduced_z)

freqs_y = np.sqrt(eigvals_y) / (2 * np.pi)
freqs_z = np.sqrt(eigvals_z) / (2 * np.pi)

# === Print frequencies ===
print("Natural frequencies (Y plane):")
for i, f in enumerate(freqs_y[:6]):
    print(f"w_y_{i+1}: {f:.2f} Hz")

print("\nNatural frequencies (Z plane):")
for i, f in enumerate(freqs_z[:6]):
    print(f"w_z_{i+1}: {f:.2f} Hz")

# === Plot mode shapes ===
x = np.linspace(0, L, n_nodes)
n_modes = min(6, len(freqs_y))

plt.figure(figsize=(10, 2.5 * n_modes))

for i in range(n_modes):
    # reconstruct full mode shape
    mode_y = np.zeros(total_dof)
    mode_z = np.zeros(total_dof)
    mode_y[free_dofs] = eigvecs_y[:, i]
    mode_z[free_dofs] = eigvecs_z[:, i]
    
    w_y = mode_y[::dof_per_node]
    w_z = mode_z[::dof_per_node]
    
    w_y /= np.max(np.abs(w_y))
    w_z /= np.max(np.abs(w_z))
    
    plt.subplot(n_modes, 1, i+1)
    plt.plot(x, w_y, '-o', label='u_y')
    plt.plot(x, w_z, '--', label='u_z')
    plt.title(f"Mode {i+1} - f_y: {freqs_y[i]:.2f} Hz, f_z: {freqs_z[i]:.2f} Hz")
    plt.xlabel("x [m]")
    plt.ylabel("Displacement (normalized)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
