import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Beam properties (can be modified)
rho = 7800          # Density / kgm^-3
L = 1.0             # Length of beam / m
D = 0.02            # Diameter of beam / m
E = 2.1e11          # Young's modulus / Nmm^-2
I = (np.pi * D**4) / 64     # Moment of inertia for circular cross-section
A = (np.pi * D**2) / 4      # Cross-sectional area

# Boundary conditions
supports = [0.0, 1.0]       # Positions of the supports in x-direction [m]

# Discretization
n_elements = 10             # Number of Nodes
n_nodes = n_elements + 1    # Number of Elements
dx = L / n_elements         # Length of one element

# Global matrices
K_global = np.zeros((2 * n_nodes, 2 * n_nodes)) # Creating a 0 Matrix with right size to save elements step by step inside
M_global = np.zeros((2 * n_nodes, 2 * n_nodes))

# Element stiffness and mass matrices (2 DOFs per node: [v, theta])
# Beam element matrices (Bernoulli beam)
def beam_element_matrices(E, I, rho, A, L):
    # Stiffness matrix
    k = E * I / L**3 * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2]
    ])
    # Consistent mass matrix
    m = rho * A * L / 420 * np.array([
        [156, 22*L, 54, -13*L],
        [22*L, 4*L**2, 13*L, -3*L**2],
        [54, 13*L, 156, -22*L],
        [-13*L, -3*L**2, -22*L, 4*L**2]
    ])
    return k, m

# Assembly of global matrices
# Explanation in Notes
for i in range(n_elements):
    k_e, m_e = beam_element_matrices(E, I, rho, A, dx)
    dof = [2*i, 2*i+1, 2*i+2, 2*i+3]
    for ii in range(4):
        for jj in range(4):
            K_global[dof[ii], dof[jj]] += k_e[ii, jj]
            M_global[dof[ii], dof[jj]] += m_e[ii, jj]

# Apply boundary conditions: displacement (v) fixed, rotation (theta) free
fixed_dofs = []
for s in supports:
    node = int(s / L * n_elements)
    fixed_dofs.append(2 * node)  # only vertical displacement is fixed

# Free DOFs
all_dofs = np.arange(2 * n_nodes)
free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

# Reduce matrices eliminating the fixed DOF 
K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
M_reduced = M_global[np.ix_(free_dofs, free_dofs)]

# Solve eigenvalue problem
eigvals, eigvecs = eigh(K_reduced, M_reduced)

# Extract first 5 natural frequencies
frequencies = np.sqrt(eigvals) / (2 * np.pi)
print("First 5 natural frequencies / Hz:")
print(frequencies[:5])

# Plot first 3 mode shapes
x = np.linspace(0, L, n_nodes)
plt.figure(figsize=(12, 8))
for i in range(3):
    mode_shape = np.zeros(n_nodes)
    counter = 0
    for j in range(n_nodes):
        if 2*j in free_dofs:
            mode_shape[j] = eigvecs[free_dofs.tolist().index(2*j), i]
        else:
            mode_shape[j] = 0  # Fixed point
    plt.subplot(3, 1, i+1)
    plt.plot(x, mode_shape, '-o')
    plt.title(f'Mode Shape {i+1}')
    plt.xlabel('Beam length = m')
    plt.ylabel('Displacement [arbitrary units]')
    plt.grid(True)
plt.tight_layout()
plt.show()
