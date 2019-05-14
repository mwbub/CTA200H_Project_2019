"""
CTA200H Computing Assignment Solutions
Author: Mathew Bub
Supervisor: Cristobal Petrovich
Date: May 17, 2019
"""
import os
import numpy as np
import matplotlib.pyplot as plt

############################
# Part 1
############################

# (i) What is tau for the current separation of the Moon?

print("Part 1(i)")
print("---------")

G = 4 * np.pi**2    # AU^3 yr^-2 Mo^-1
m_sun = 1           # Mo (solar masses)
m_earth = 3.003e-6  # Mo
r_pl = 1            # AU
a_moon = 1/388.6    # AU
P_moon = 2 * np.pi * (a_moon**3 / G / m_earth)**0.5               # Years
tau = m_earth / m_sun * r_pl**3 / a_moon**3 * P_moon / 3 / np.pi  # Years

print("tau = {:.3f} years".format(tau))

# (ii) Implement a Runge-Kutta of order 4 in Python to solve the ODEs

print("\nPart 1(ii)")
print("----------")

def derivatives(x, t):
    """
    Calculate de/dt and dj/dt at a point (x, t), where x = (e, j).
    """
    # Extract e and j from x
    e = x[:3]
    j = x[3:]
    
    # Earth position vector
    r_pl = np.array([np.cos(t), np.sin(t), 0])
    
    # Cross products
    j_cross_e = np.cross(j, e)
    j_cross_r = np.cross(j, r_pl)
    e_cross_r = np.cross(e, r_pl)
    
    # Dot products
    r_dot_e = np.dot(r_pl, e)
    r_dot_j = np.dot(r_pl, j)
    
    # Derivatives
    de = -(2 * j_cross_e - 5 * r_dot_e * j_cross_r + r_dot_j * e_cross_r) / tau
    dj = -(r_dot_j * j_cross_r - 5 * r_dot_e * e_cross_r) / tau
    
    return np.hstack([de, dj])
    
def step(x, t, dt):
    """
    Increment e and j by dt using a 4th order Runge-Kutta integration.
    """
    # Calculate the 4th order Runge-Kutta derivatives
    m = derivatives(x, t)
    n = derivatives(x + m * dt/2, t + dt/2)
    p = derivatives(x + n * dt/2, t + dt/2)
    q = derivatives(x + p * dt, t + dt)
    
    return x + (m + 2*n + 2*p + q)/6 * dt

def e_j_from_angles(Omega, omega, I, e0):
    e = e0 * np.array([np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(I),
                       np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(I),
                       np.sin(I) * np.sin(omega)])
    j = (1 - e0**2)**0.5 * np.array([np.sin(I) * np.sin(Omega),
                                     -np.sin(I) * np.cos(Omega),
                                     np.cos(I)])
    return e, j

# Initial conditions
Omega = 0
omega = np.pi/2
I = np.pi/3
e0 = 0.05

e, j = e_j_from_angles(Omega, omega, I, e0)

# Initial values of e and j
x = np.hstack([e, j])

# Set up integration parameters and arrays
dt = 1/20                       # Years
ts = np.arange(0, 10*tau, dt)   # Time steps
es = np.zeros((len(ts), 3))     # e vectors
js = np.zeros((len(ts), 3))     # j vectors

# Values of e and j at t=0
es[0] = e
js[0] = j

# Perform the integration
for i in range(1, len(ts)):
    x = step(x, ts[i-1], ts[i] - ts[i-1])
    es[i] = x[:3]
    js[i] = x[3:]
    
# Sanity checks
norms = np.sum(es**2 + js**2, axis=1)

print("\nSanity Checks")
print("-------------")
print("Minimum norm = {}".format(np.min(norms)))
print("Maximum norm = {}".format(np.max(norms)))

dots = np.array([np.dot(es[i], js[i]) for i in range(len(es))])
print("Minimum dot product = {}".format(np.min(np.abs(dots))))
print("Maximum dot product = {}".format(np.max(dots)))

############################
# Part 2
############################

ex, ey, ez = es.T
jx, jy, jz = js.T
e = (ex**2 + ey**2 + ez**2)**0.5

I = np.arctan2((jx**2 + jy**2)**0.5, jz)
Omega = np.arctan2(jx, -jy)
omega = np.arctan2(-ex * np.sin(Omega) + ey * np.cos(Omega) * np.cos(I) + ez * np.sin(I),
                   ex * np.cos(Omega) + ey * np.sin(Omega))

jz2 = (1 - e**2)**0.5 * np.cos(I)

if not os.path.exists("plots"):
    os.mkdir("plots")

plt.plot(ts, e)
plt.xlabel("Time (years)")
plt.ylabel("$e$")
plt.savefig("plots/e.pdf")
plt.show()

plt.plot(ts, I * 180 / np.pi)
plt.xlabel("Time (years)")
plt.ylabel("I (deg)")
plt.savefig("plots/I.pdf")
plt.show()

plt.plot(ts, jz)
plt.xlabel("Time (years)")
plt.ylabel("$j_z$")
plt.savefig("plots/j_z.pdf")
plt.show()

plt.plot(ts, jz2)
plt.xlabel("Time (years)")
plt.ylabel("$j_z$")
plt.show()

plt.plot(e, omega * 180 / np.pi)
plt.xlabel("e")
plt.ylabel("$\omega$ (deg)")
plt.savefig("plots/omega_e.pdf")
plt.show()
    