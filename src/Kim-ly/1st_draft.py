import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Beam and material properties
L = 1.0                     # Total beam length [m]
E = 210e9                   # Young's modulus [Pa]
rho = 7800                  # Density [kg/m^3]
D = 0.02                    # Diameter of beam / m
I = (np.pi * D**4) / 64     # Moment of inertia for circular cross-section [m^4]
A = (np.pi * D**2) / 4      # Cross-sectional area [m^2]




# === FEM discretization ===
n_elem = 10
n_nodes = n_elem + 1
dof_per_node = 2                        # 2 DOFs: Transverse displacement & angle
total_dof = dof_per_node * n_nodes
dx = L / n_elem                         # Length of one element

# Element stiffness and mass matrices (4x4)
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

# Initialize global matrices 
K_global = np.zeros((total_dof, total_dof)) # Creating a 0 Matrix with right size to save elements step by step inside
M_global = np.zeros((total_dof, total_dof))


# Assembly of global matrices
# Explanation in Notes
for e in range(n_elem):
    k_e, m_e = beam_element_matrices(E, I, rho, A, dx)
    dof_map = [2*e, 2*e+1, 2*e+2, 2*e+3]
    
    for i in range(4):
        for j in range(4):
            K_global[dof_map[i], dof_map[j]] += k_e[i, j]
            M_global[dof_map[i], dof_map[j]] += m_e[i, j]
            

# Boundary conditions (Simply supported beam: w=0 at both ends)
# Remove DOFs: displacement DOF at node 0 and node n_nodes-1
constrained_dofs = [0, 2*n_nodes - 2]                               # w at both ends
free_dofs = np.setdiff1d(np.arange(total_dof), constrained_dofs)

# Reduce matrices eliminating the fixed DOF 
K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
M_reduced = M_global[np.ix_(free_dofs, free_dofs)]

# Solve eigenvalue problem
eigvals, eigvecs = eigh(K_reduced, M_reduced)

#                    Filter small/negative eigenvalues (numerical noise)
#                    eigvals = eigvals[eigvals > 1e-8]
numerical_freqs = np.sqrt(eigvals) / (2 * np.pi)

# Print first natural frequencies
print("Numerical Natural frequencies (Hz):")
for i, f in enumerate(numerical_freqs[:5]):
    print(f"Mode {i+1}: {f:.2f} Hz")
    
    
# Plot mode shapes (displacement part only)
x = np.linspace(0, L, n_nodes)
n_modes = min(5, len(numerical_freqs))            # Change number of mode shapes to plot here

plt.figure(figsize=(10, 8))
for i in range(n_modes):
    # Reconstruct full mode shape including constrained DOFs
    full_mode = np.zeros(total_dof)
    full_mode[free_dofs] = eigvecs[:, i]

    w_only = full_mode[::2]  # Take displacement DOFs only
    w_only /= np.max(np.abs(w_only))  # Normalize

    plt.subplot(n_modes, 1, i+1)
    plt.plot(x, w_only, '-o')
    plt.title(f"Mode Shape {i+1} - {numerical_freqs[i]:.2f} Hz")
    plt.xlabel(f"Beam length = {L} m")
    plt.ylabel("Displacement ")
    plt.grid(True)

plt.tight_layout()
plt.show()




# === ANALYTICAL SOLUTION for Simply Supported Beam ===
def beta_n(n, L):
    return n * np.pi / L

def omega_n(n, E, I, rho, A, L):
    beta = beta_n(n, L)
    return beta**2 * np.sqrt(E * I / (rho * A))

def mode_shape(n, x, L):
    return np.sin(beta_n(n, L) * x)

# === Calculate Natural Frequencies and Mode Shapes ===
n_modes = 5
x_vals = np.linspace(0, L, 500)

analytical_freqs = []
analytical_shapes = []

for n in range(1, n_modes + 1):
    omega = omega_n(n, E, I, rho, A, L)
    f = omega / (2 * np.pi)
    analytical_freqs.append(f)
    if n <= 3:  # Only plot first 3 mode shapes
        shape = mode_shape(n, x_vals, L)
        analytical_shapes.append(shape)

# === OUTPUT ===
print("\nAnalytical Natural Frequencies (Hz):")
for i, f in enumerate(analytical_freqs):
    print(f"Mode {i+1}: {f:.2f} Hz")

# === PLOT First 3 Analytical Mode Shapes ===
plt.figure(figsize=(10, 6))
for i, shape in enumerate(analytical_shapes):
    plt.plot(x_vals, shape, label=f"Mode {i+1}")
plt.title("First 3 Analytical Mode Shapes of a Simply Supported Beam")
plt.xlabel("Beam Length x [m]")
plt.ylabel("Displacement (normalized)")
plt.grid(True)
plt.legend()
plt.show()





# === COMPARISON WITH NUMERICAL FREQUENCIES ===
print("\nComparison of Analytical vs Numerical Natural Frequencies:")
print(f"{'Mode':<5} {'Analytical (Hz)':<20} {'Numerical (Hz)':<20} {'Error (%)':<10}")
for i in range(n_modes):
    ana = analytical_freqs[i]
    num = numerical_freqs[i]
    error = abs((num - ana) / ana) * 100
    print(f"{i+1:<5} {ana:<20.2f} {num:<20.2f} {error:<10.2f}")
