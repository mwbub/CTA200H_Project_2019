"""
CTA200H Computing Assignment Solutions
Author: Mathew Bub
Supervisor: Cristobal Petrovich
Date: May 17, 2019
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Use TeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

############################
# Part 1
############################

# (i) What is tau for the current separation of the Moon?

print("Part 1(i)")
print("---------")

# Calculation of tau_moon
G = 4 * np.pi**2    # AU^3 yr^-2 Mo^-1
m_sun = 1           # Mo (solar masses)
m_earth = 3.003e-6  # Mo
r_pl = 1            # AU
a_moon = 1/388.6    # AU
P_moon = 2 * np.pi * (a_moon**3 / G / m_earth)**0.5               # Years
tau = m_earth / m_sun * r_pl**3 / a_moon**3 * P_moon / 3 / np.pi  # Years

# Recalculate tau for a closer Moon
a_moon_close = a_moon / 10
P_moon_close = 2 * np.pi * (a_moon_close**3 / G / m_earth)**0.5
tau_close = m_earth / m_sun * r_pl**3 / a_moon_close**3 * P_moon_close / 3 / np.pi

print("tau = {:.3f} years".format(tau))
print("tau_close = {:.3f} years".format(tau_close))

# (ii) Implement a Runge-Kutta of order 4 in Python to solve the ODEs

print("\nPart 1(ii)")
print("----------")
    
def step(f, x, t, dt):
    """
    Increment x(t) by dt using a 4th order Runge-Kutta integration, given
    that x' = f(x, t)
    """
    # Calculate the 4th order Runge-Kutta derivatives
    m = f(x, t)
    n = f(x + m * dt/2, t + dt/2)
    p = f(x + n * dt/2, t + dt/2)
    q = f(x + p * dt, t + dt)
    
    return x + (m + 2*n + 2*p + q)/6 * dt

def integrate(f, x0, tmin, tmax, dt):
    """
    Integrate x' = f(x, t) from tmin to tmax with initial data x0 and step dt.
    """
    ts = np.arange(tmin, tmax, dt)
    xs = np.zeros((len(ts), len(x0)))
    x = np.copy(x0)
    xs[0] = x
    
    for i in range(1, len(ts)):
        x = step(f, x, ts[i-1], dt)
        xs[i] = x
        
    return xs, ts

def dej_dt(x, t, tau):
    """
    Calculate de/dt and dj/dt at a point (x, t), where x = (e, j).
    """
    # Extract e and j from x
    e = x[:3]
    j = x[3:]
    
    # Earth position vector
    r_pl = np.array([np.cos(2*np.pi*t), np.sin(2*np.pi*t), 0])
    
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
    
    # Recombine the derivatives into a single array
    return np.hstack([de, dj])

def e_j_from_angles(Omega, omega, I, e0):
    """
    Calculate the e and j vectors from the orbital elements.
    """
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

# Initial values of e and j
e, j = e_j_from_angles(Omega, omega, I, e0)
x = np.hstack([e, j])

dt = 1/20 # Years

# Perform the integration
xs, ts = integrate(lambda x, t: dej_dt(x, t, tau), x, 0, 10 * tau, dt)
es = xs[:,:3]
js = xs[:,3:]
    
# Sanity checks
norms = np.sum(es**2 + js**2, axis=1)
dots = np.array([np.dot(es[i], js[i]) for i in range(len(es))])

print("\nSanity Checks")
print("-------------")
print("Minimum norm = {}".format(np.min(norms)))
print("Maximum norm = {}".format(np.max(norms)))
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

if not os.path.exists("plots"):
    os.mkdir("plots")
    
fig, ax = plt.subplots(2, 2, figsize=(10,9))

ax[0,0].plot(ts, e)
ax[0,0].set_xlabel("Time (years)")
ax[0,0].set_ylabel("$e$")
ax[0,0].hlines((1 - 5/3 * np.cos(np.pi/3)**2)**0.5, 0, 10*tau)
ax[0,0].set_xlim(0, 10*tau)
ax[0,0].text(0.75, 0.9, '$e_\mathrm{max}$', 
             horizontalalignment='center', 
             verticalalignment='center', 
             transform = ax[0,0].transAxes,
             fontsize=14)
ax[0,0].text(0, 1.075, '$\mathbf{(a)}$', 
             horizontalalignment='center', 
             verticalalignment='center', 
             transform = ax[0,0].transAxes,
             fontsize=20)

ax[0,1].plot(ts, I * 180 / np.pi)
ax[0,1].set_xlabel("Time (years)")
ax[0,1].set_ylabel("$I$ (deg)")
ax[0,1].set_xlim(0, 10*tau)
ax[0,1].text(0, 1.075, '$\mathbf{(b)}$', 
             horizontalalignment='center', 
             verticalalignment='center', 
             transform = ax[0,1].transAxes,
             fontsize=20)

ax[1,0].plot(ts, jz)
ax[1,0].set_xlabel("Time (years)")
ax[1,0].set_ylabel("$j_z$")
ax[1,0].set_xlim(0, 10*tau)
ax[1,0].text(0, 1.075, '$\mathbf{(c)}$', 
             horizontalalignment='center', 
             verticalalignment='center', 
             transform = ax[1,0].transAxes,
             fontsize=20)

ax[1,1].plot(omega * 180 / np.pi, e)
ax[1,1].set_xlabel("$\omega$ (deg)")
ax[1,1].set_ylabel("$e$")
ax[1,1].text(0, 1.075, '$\mathbf{(d)}$', 
             horizontalalignment='center', 
             verticalalignment='center', 
             transform = ax[1,1].transAxes,
             fontsize=20)

plt.savefig("plots/figure1.pdf", bbox_inches="tight")
plt.show()
    