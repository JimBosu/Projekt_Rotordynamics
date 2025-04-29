import numpy as np
import matplotlib.pyplot as plt

# === PARAMETER ===
rho = 7800          # Density [kg/m^3]
L = 1.0             # Length of beam [m]
D = 0.02            # Diameter of beam [m]
E = 2.1e11          # Young's modulus [Pa]
I = (np.pi * D**4) / 64  # Moment of inertia for circular cross-section
A = (np.pi * D**2) / 4   # Cross-sectional area

# Lagerpositionen (einfach veränderbar):
x_supports = [0.0, 1]  # Positionen der Lager in x-Richtung [m]

# === ANALYTISCHE LÖSUNG für zweiseitig gelagerten Balken ===
def beta_n(n, L):
    return n * np.pi / L

def omega_n(n, E, I, rho, A, L):
    beta = beta_n(n, L)
    return beta**2 * np.sqrt(E * I / (rho * A))

def mode_shape(n, x, L):
    return np.sin(beta_n(n, L) * x)

# === Eigenfrequenzen und Moden berechnen ===
n_modes = 5
x_vals = np.linspace(0, L, 500)

frequencies = []
mode_shapes = []

for n in range(1, n_modes + 1):
    omega = omega_n(n, E, I, rho, A, L)
    f = omega / (2 * np.pi)
    frequencies.append(f)
    if n <= 3:
        shape = mode_shape(n, x_vals, L)
        mode_shapes.append(shape)

# === AUSGABE ===
print("Erste 5 Eigenfrequenzen (Hz):")
for i, f in enumerate(frequencies):
    print(f"f{i+1} = {f:.2f} Hz")

# === PLOTTEN der ersten 3 Moden ===
plt.figure(figsize=(10, 6))
for i, shape in enumerate(mode_shapes):
    plt.plot(x_vals, shape, label=f"Mode {i+1}")

plt.title("Erste 3 Eigenmoden eines zweiseitig gelagerten Balkens")
plt.xlabel("Balkenlänge x [m]")
plt.ylabel("Auslenkung (normiert)")
plt.grid(True)
plt.legend()
plt.show()
