import sympy as sp
import pylbm
import numpy as np
from mpi4py import MPI

#get MPI info
comm=MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

#geometry
dx = 0.05
dt = 0.005
la = dx/dt
L_box = 1.

#characteristic dimensions
U = 1.
L = dx
T = dt

#geometry in lattice units
dx_nd = dx/L
L_box_nd = L_box/L
la_nd = 1.

#material parameters (actual values and in lattice units)
E = 0.1/0.1731
E_nd = T/(L*L)*E
nu = .7
K = E/(2*(1-nu))
K_nd = K*T/L**2
mu = E/(2*(1+nu))
mu_nd = mu*T/L**2
theta = 1/3

#define symbols
X, Y = sp.symbols('X, Y')
u, v, LA = sp.symbols('u, v, LA')
THETA, MU_ND, K_ND, GAMMA = sp.symbols('THETA, MU_ND, K_ND, GAMMA')

#moment matrix
M = sp.Matrix([[1,0,-1,0,1,-1,-1,1],[0,1,0,-1,1,1,-1,-1],[0,0,0,0,1,-1,1,-1],[1,1,1,1,2,2,2,2],
    [1,-1,1,-1,0,0,0,0],[0,0,0,0,1,-1,-1,1],[0,0,0,0,1,1,-1,-1],[GAMMA,GAMMA,GAMMA,GAMMA,1+2*GAMMA,1+2*GAMMA,1+2*GAMMA,1+2*GAMMA]])

#equilibrium moments in order
Meq = [u,v,0,0,0,THETA*u,THETA*v,0]

#relaxation parameters
w10 = 0.
w01 = 0.
w11 = 1/(MU_ND/THETA+.5)
ws = 1/(2*K_ND/(1+THETA)+.5)
wd = 1/(2*MU_ND/(1-THETA)+.5)
w12 = 1.5
w21 = 1.5
wf = 1.
omega = [w10,w01,w11,ws,wd,w12,w21,wf]
gamma = theta*.5/((1+theta)*((1/ws-.5).evalf(subs={K_ND:K_nd,THETA:theta})-.5))

#forcing
g_x = 1e-6*(sp.sin(2*sp.pi*X*L)+sp.cos(2*sp.pi*Y*L))
g_y = -1e-6*(sp.cos(2*sp.pi*X*L)+sp.sin(2*sp.pi*Y*L))

#initial condition
def u_init(x,y):
    return 0.

def v_init(x,y):
    return 0.

dico = {
    'box' : {'x': [0,L_box_nd], 'y': [0,L_box_nd], 'label': -1},
    'space_step': dx_nd,
    'scheme_velocity': LA,
    'parameters': {LA: la_nd,
            THETA: theta,
            MU_ND: mu_nd,
            K_ND: K_nd,
            GAMMA: gamma,
            },
    'init': {u: u_init, v: v_init},
    'generator': 'cython',
    'lbm_algorithm': {'name': pylbm.algorithm.BaseAlgorithm},
    'schemes':[
            {
                'velocities': list(range(1,9)),
                'conserved_moments': [u,v],
                'M': M,
                'equilibrium': Meq,
                'relaxation_parameters': omega,
                'source_terms': {u: g_x, v: g_y}
                }]
    }

#create and run the simulation
sim = pylbm.Simulation(dico)

for i in range(10000): 
    sim.one_time_step()

#function handles
g_x_f = sp.lambdify([X,Y],g_x,'numpy')
g_y_f = sp.lambdify([X,Y],g_y,'numpy')

#meshgrid coordinates
x,y = np.meshgrid(sim.domain.x,sim.domain.y,indexing='ij')
#the indexing='ij' option is important here to ensure the forcing array and moment array are ordered the same way

#this piece of code essentially gives x.flatten() and y.flatten() I think
mask = (sim.domain.in_or_out == sim.domain.valin)
ind = np.where(mask)
coords = np.zeros((ind[0].size, 3))
for i in range(sim.domain.dim):
    coords[:, i] = sim.domain.coords_halo[i][ind[i]]
x_flat = coords[:,0]
y_flat = coords[:,1]
    
#displacements
u = sim.m[0] - 0.5*g_x_f(x, y)
v = sim.m[1] - 0.5*g_y_f(x, y)
   
#numerical relaxation rates
w11_sub = w11.evalf(subs=dico['parameters'])
ws_sub = ws.evalf(subs=dico['parameters'])
wd_sub = wd.evalf(subs=dico['parameters'])

#bared moments
m_bar_11 = 0.5*sim.m[2]*(1/(1-w11_sub)+1)
m_bar_s = 0.5*sim.m[3]*(1/(1-ws_sub)+1) 
m_bar_d = 0.5*sim.m[4]*(1/(1-wd_sub)+1)

#stresses
sigma_xx = -0.5*(m_bar_s+m_bar_d)
sigma_xy = -m_bar_11
sigma_yy = -0.5*(m_bar_s-m_bar_d)

#strains
eps_xx = -0.25*(m_bar_s/K_nd+m_bar_d/mu_nd)
eps_xy = -0.5*m_bar_11/mu_nd
eps_yy = -0.25*(m_bar_s/K_nd-m_bar_d/mu_nd)

#write data to files; since we don't want to deal with the exact shapes of all domains created by MPI, we will flatten all output. No geometry information is
#lost as we still have the coordinates for every node followed by all solution quantities at that node. 
write = np.column_stack((x_flat,y_flat,u.flatten(),v.flatten(),sigma_xx.flatten(),sigma_xy.flatten(),sigma_yy.flatten(),eps_xx.flatten(),eps_xy.flatten(),eps_yy.flatten()))
np.savetxt('./results/data_'+str(myrank)+'.out',write)