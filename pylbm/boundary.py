# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Module for LBM boundary conditions
"""

import collections
import logging
import types
import numpy as np
from sympy import symbols, IndexedBase, Idx, Eq

from .storage import Array

log = logging.getLogger(__name__) #pylint: disable=invalid-name

#pylint: disable=too-few-public-methods
class BoundaryVelocity:
    """
    Indices and distances for the label and the velocity ksym
    """
    def __init__(self, domain, label, ksym):
        # We are looking for the points on the outside that have a speed
        # that goes in (index ksym) on a border labeled by label.
        # We go through all the lattice velocities and determine the inner points
        # that have the symmetric lattice velocity (index k) that comes out
        # then we write in a list with the order of the lattice velocities
        # involved in the schemes:
        # - indices of the corresponding external points
        # - associated distances
        self.label = label
        self.v = domain.stencil.unique_velocities[ksym]
        v = self.v.get_symmetric()
        num = domain.stencil.unum2index[v.num]

        ind = np.where(domain.flag[num] == self.label)
        self.indices = np.array(ind)
        if self.indices.size != 0:
            self.indices += np.asarray(v.v)[:, np.newaxis]
        self.distance = np.array(domain.distance[(num,) + ind])
        self.normal = np.array(domain.normal[(num,) + ind])  #

class Boundary:
    """
    Construct the boundary problem by defining the list of indices on the border and the methods used on each label.

    Parameters
    ----------
    domain : pylbm.Domain
        the simulation domain
    dico : dictionary
        describes the boundaries
            - key is a label
            - value are again a dictionnary with
                + "method" key that gives the boundary method class used (Bounce_back, Anti_bounce_back, ...)
                + "value_bc" key that gives the value on the boundary

    Attributes
    ----------
    bv_per_label : dictionnary
        for each label key, a list of spatial indices and distance define for each velocity the points
        on the domain that are on the boundary.

    methods : list
        list of boundary methods used in the LBM scheme
        The list contains Boundary_method instance.

    """
    #pylint: disable=too-many-locals
    def __init__(self, domain, generator, dico):
        self.domain = domain

        # build the list of indices for each unique velocity and for each label
        self.bv_per_label = {}
        for label in self.domain.list_of_labels():
            if label in [-1, -2]: # periodic or interface conditions
                continue
            dummy_bv = []
            for k in range(self.domain.stencil.unvtot):
                dummy_bv.append(BoundaryVelocity(self.domain, label, k))
            self.bv_per_label[label] = dummy_bv

        # build the list of boundary informations for each stencil and each label
        dico_bound = dico.get('boundary_conditions', {})
        stencil = self.domain.stencil

        istore = collections.OrderedDict() # important to set the boundary conditions always in the same way !!!
        ilabel = {}
        distance = {}
        value_bc = {}
        time_bc = {}
        normal = {}


        #pylint: disable=too-many-nested-blocks
        for label in self.domain.list_of_labels():
            if label in [-1, -2]: # periodic or interface conditions
                continue

            value_bc[label] = dico_bound[label].get('value', None)
            time_bc[label] = dico_bound[label].get('time_bc', False)
            methods = dico_bound[label]['method']
            # for each method get the list of points, the labels and the distances
            # where the distribution function must be updated on the boundary
            for k, v in methods.items():
                for inumk, numk in enumerate(stencil.num[k]):
                    if self.bv_per_label[label][stencil.unum2index[numk]].indices.size != 0:
                        indices = self.bv_per_label[label][stencil.unum2index[numk]].indices
                        distance_tmp = self.bv_per_label[label][stencil.unum2index[numk]].distance
                        normal_tmp = self.bv_per_label[label][stencil.unum2index[numk]].normal
                        velocity = (inumk + stencil.nv_ptr[k])*np.ones(indices.shape[1], dtype=np.int32)[np.newaxis, :]
                        ilabel_tmp = label*np.ones(indices.shape[1], dtype=np.int32)
                        istore_tmp = np.concatenate([velocity, indices])
                        if istore.get(v, None) is None:
                            istore[v] = istore_tmp.copy()
                            ilabel[v] = ilabel_tmp.copy()
                            distance[v] = distance_tmp.copy()
                            normal[v] = normal_tmp.copy()
                        else:
                            istore[v] = np.concatenate([istore[v], istore_tmp], axis=1)
                            ilabel[v] = np.concatenate([ilabel[v], ilabel_tmp])
                            distance[v] = np.concatenate([distance[v], distance_tmp])
                            normal[v] = np.concatenate([normal[v], normal_tmp])  #

        # for each method create the instance associated
        self.methods = []
        for k in list(istore.keys()):
            #print(k)
            self.methods.append(k(istore[k], ilabel[k], distance[k], normal[k], stencil,
                                  value_bc, time_bc, domain.distance.shape, generator))


#pylint: disable=protected-access
class BoundaryMethod:
    """
    Set boundary method.

    Parameters
    ----------
    FIXME : add parameters documentation

    Attributes
    ----------
    feq : ndarray
        the equilibrium values of the distribution function on the border
    rhs : ndarray
        the additional terms to fix the boundary values
    distance : ndarray
        distance to the border (needed for Bouzidi type conditions)
    istore : ndarray
        indices of points where we store the boundary condition
    ilabel : ndarray
        label of the boundary
    iload : list
        indices of points needed to compute the boundary condition
    value_bc : dictionnary
       the prescribed values on the border

    """
    def __init__(self, istore, ilabel, distance, normal, stencil, value_bc, time_bc, nspace, generator):
        self.istore = istore
        self.feq = np.zeros((stencil.nv_ptr[-1], istore.shape[1]))
        self.rhs = np.zeros(istore.shape[1])
        self.ilabel = ilabel
        self.distance = distance
        self.normal = normal
        self.stencil = stencil
        self.time_bc = {}
        self.value_bc = {}
        for k in np.unique(self.ilabel):
            self.value_bc[k] = value_bc[k]
            self.time_bc[k] = time_bc[k]
        self.iload = []
        self.nspace = nspace
        self.generator = generator

        # used if time boundary
        self.func = []
        self.args = []
        self.f = []
        self.m = []
        self.indices = []

    def fix_iload(self):
        """
        Transpose iload and istore.

        Must be fix in a future version.
        """
        # Fixme : store in a good way and in the right type istore and iload
        for i in range(len(self.iload)):
            self.iload[i] = np.ascontiguousarray(self.iload[i].T, dtype=np.int32)
        self.istore = np.ascontiguousarray(self.istore.T, dtype=np.int32)

    #pylint: disable=too-many-locals
    def prepare_rhs(self, simulation):
        """
        Compute the distribution function at the equilibrium with the value on the border.

        Parameters
        ----------
        simulation : Simulation
            simulation class

        """

        nv = simulation.container.nv
        sorder = simulation.container.sorder
        nspace = [1]*(len(sorder)-1)
        v = self.stencil.get_all_velocities()

        gpu_support = simulation.container.gpu_support

        for key, value in self.value_bc.items():
            if value is not None:
                indices = np.where(self.ilabel == key)
                # TODO: check the index in sorder to be the most contiguous
                nspace[0] = indices[0].size
                k = self.istore[0, indices]

                s = 1 - self.distance[indices]
                coords = tuple()
                for i in range(simulation.domain.dim):
                    x = simulation.domain.coords_halo[i][self.istore[i + 1, indices]]
                    x += s*v[k, i]*simulation.domain.dx
                    x = x.ravel()
                    for j in range(1, simulation.domain.dim): #pylint: disable=unused-variable
                        x = x[:, np.newaxis]
                    coords += (x,)

                m = Array(nv, nspace, 0, sorder, gpu_support=gpu_support)
                m.set_conserved_moments(simulation.scheme.consm)

                f = Array(nv, nspace, 0, sorder, gpu_support=gpu_support)
                f.set_conserved_moments(simulation.scheme.consm)

                args = coords
                if isinstance(value, types.FunctionType):
                    func = value
                elif isinstance(value, tuple):
                    func = value[0]
                    args += value[1]

                if self.time_bc[key]:
                    func(f, m, 0, *args)
                else:
                    func(f, m, *args)

                simulation.equilibrium(m)
                simulation.m2f(m, f)

                if self.generator.backend.upper() == "LOOPY":
                    f.array_cpu[...] = f.array.get()
                
                self.feq[:, indices[0]] = f.swaparray.reshape((nv, indices[0].size))

                if self.time_bc[key]:
                    self.func.append(func)
                    self.args.append(args)
                    self.f.append(f)
                    self.m.append(m)
                    self.indices.append(indices[0])

    def update_feq(self, simulation):
        t = simulation.t
        nv = simulation.container.nv

        for i in range(len(self.func)):
            self.func[i](self.f[i], self.m[i], t, *self.args[i])
            simulation.equilibrium(self.m[i])
            simulation.m2f(self.m[i], self.f[i])

            if self.generator.backend.upper() == "LOOPY":
                self.f[i].array_cpu[...] = self.f[i].array.get()

            self.feq[:, self.indices[i]] = self.f[i].swaparray.reshape((nv, self.indices[i].size))

    def _get_istore_iload_symb(self, dim):
        ncond = symbols('ncond', integer=True)

        istore = symbols('istore', integer=True)
        istore = IndexedBase(istore, [ncond, dim+1])

        iload = []
        for i in range(len(self.iload)):
            iloads = symbols('iload%d'%i, integer=True)
            iload.append(IndexedBase(iloads, [ncond, dim+1]))
        return istore, iload, ncond

    @staticmethod
    def _get_rhs_dist_symb(ncond):
        rhs = IndexedBase('rhs', [ncond])
        dist = IndexedBase('dist', [ncond])
        return rhs, dist

    def update(self, ff, **kwargs):
        """
        Update distribution functions with this boundary condition.

        Parameters
        ----------

        ff : array
            The distribution functions
        """
        from .symbolic import call_genfunction

        args = self._get_args(ff)
        args.update(kwargs)
        call_genfunction(self.function, args) #pylint: disable=no-member

    #pylint: disable=possibly-unused-variable
    def _get_args(self, ff):
        dim = len(ff.nspace)
        nx = ff.nspace[0]
        if dim > 1:
            ny = ff.nspace[1]
        if dim > 2:
            nz = ff.nspace[2]

        f = ff.array

        for i in range(len(self.iload)):
            exec('iload{i} = self.iload[{i}]'.format(i=i)) #pylint: disable=exec-used

        istore = self.istore
        rhs = self.rhs
        if hasattr(self, 's'):
            dist = self.s
        ncond = istore.shape[0]
        return locals()

    def move2gpu(self):
        """
        Move arrays needed to compute the boundary on the GPU memory.
        """
        if self.generator.backend.upper() == "LOOPY":
            try:
                import pyopencl as cl
                import pyopencl.array #pylint: disable=unused-variable
                from .context import queue
            except ImportError:
                raise ImportError("Please install loo.py")

            self.rhs = cl.array.to_device(queue, self.rhs)
            if hasattr(self, 's'):
                self.s = cl.array.to_device(queue, self.s) #pylint: disable=attribute-defined-outside-init
            self.istore = cl.array.to_device(queue, self.istore)
            for i in range(len(self.iload)):
                self.iload[i] = cl.array.to_device(queue, self.iload[i])

class BounceBack(BoundaryMethod):
    """
    Boundary condition of type bounce-back

    Notes
    ------

    .. plot:: codes/bounce_back.py

    """
    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k][np.newaxis, :]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:] + v[k].T
        self.iload.append(np.concatenate([ksym, indices]))

    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.feq[k, np.arange(k.size)] - self.feq[ksym, np.arange(k.size)]

    #pylint: disable=too-many-locals
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)
        rhs, _ = self._get_rhs_dist_symb(ncond)

        idx = Idx(ix, (0, ncond))
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        fload = indexed('f', [ns, nx, ny, nz], index=[iload[0][idx, k] for k in range(dim+1)], priority=sorder)

        self.generator.add_routine(('bounce_back', For(idx, Eq(fstore, fload + rhs[idx]))))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.bounce_back

class LEDirichlet2D(BoundaryMethod):
    """
    Boundary condition of type Dirichlet for 2D Linear Elasticity

    Notes
    ------

    .. plot:: codes/bounce_back.py

    """
    
    def __init__(self, istore, ilabel, distance, normal, stencil, value_bc, time_bc, nspace, generator):
        super(LEDirichlet2D, self).__init__(istore, ilabel, distance, normal, stencil, value_bc, time_bc, nspace, generator)
        self.s_1 = np.zeros(self.istore.shape[1])
        self.s_1_old = np.zeros(self.istore.shape[1])
        self.s_0 = np.zeros(self.istore.shape[1])
        self.w_11 = 0.
        self.w_s = 0.
        self.w_d = 0.
        self.k_nd = 0.
        self.mu_nd = 0.
        self.theta = 0.


    def prepare_rhs(self, simulation):
        """
        Compute the distribution function at the equilibrium with the value on the border.

        Parameters
        ----------
        simulation : Simulation
            simulation class

        """
        #Initialize Simulation-Parameters for use in update_feq()

        #relaxation rates
        self.w_11 = float(simulation.scheme.s.evalf(subs=simulation.scheme.param)[2])
        self.w_s = float(simulation.scheme.s.evalf(subs=simulation.scheme.param)[3])
        self.w_d = float(simulation.scheme.s.evalf(subs=simulation.scheme.param)[4])
        
        #material properties
        self.k_nd = simulation.scheme.param[symbols('K_ND')]
        self.mu_nd = simulation.scheme.param[symbols('MU_ND')]
        self.theta = simulation.scheme.param[symbols('THETA')]
        
        #copy f to use with one time step delay
        self.f_delay = simulation.container.F.array.copy()

        nv = simulation.container.nv
        sorder = simulation.container.sorder
        nspace = [1]*(len(sorder)-1)
        v = self.stencil.get_all_velocities()

        gpu_support = simulation.container.gpu_support

        for key, value in self.value_bc.items():
            if value is not None:
                indices = np.where(self.ilabel == key)
                # TODO: check the index in sorder to be the most contiguous
                nspace[0] = indices[0].size
                k = self.istore[0, indices]

                s = 1 - self.distance[indices]
                coords = tuple()
                for i in range(simulation.domain.dim):
                    x = simulation.domain.coords_halo[i][self.istore[i + 1, indices]]
                    x += s*v[k, i]*simulation.domain.dx
                    x = x.ravel()
                    for j in range(1, simulation.domain.dim): #pylint: disable=unused-variable
                        x = x[:, np.newaxis]
                    coords += (x,)

                m = Array(nv, nspace, 0, sorder, gpu_support=gpu_support)
                m.set_conserved_moments(simulation.scheme.consm)

                f = Array(nv, nspace, 0, sorder, gpu_support=gpu_support)
                f.set_conserved_moments(simulation.scheme.consm)

                args = coords
                if isinstance(value, types.FunctionType):
                    func = value
                elif isinstance(value, tuple):
                    func = value[0]
                    args += value[1]

                if self.time_bc[key]:
                    func(f, m, 0, *args)
                else:
                    func(f, m, *args)
                
                k = self.iload[0][0,:]
                ksym = self.stencil.get_symmetric()[k]
                m_index = 0
                for icond in indices[0]:
                    v_i = ksym[icond]
                    if v_i <= 3:
                        self.s_0[icond] = (1-self.theta)*(v[v_i][0]*m[0,m_index]+v[v_i][1]*m[1,m_index])
                    else:
                        self.s_0[icond] = .5*self.theta*(v[v_i][0]*m[0,m_index]+v[v_i][1]*m[1,m_index])
                    m_index+=1            

                if self.generator.backend.upper() == "LOOPY":
                    f.array_cpu[...] = f.array.get()

                self.feq[:, indices[0]] = f.swaparray.reshape((nv, indices[0].size))

                if self.time_bc[key]:
                    self.func.append(func)
                    self.args.append(args)
                    self.f.append(f)
                    self.m.append(m)
                    self.indices.append(indices[0])

    def update_feq(self,simulation):
        """
        override this method to access simulation.m 
        """
        t = simulation.t
        nv = simulation.container.nv

        for i in range(len(self.func)):
            self.func[i](self.f[i], self.m[i], t, *self.args[i])
            simulation.equilibrium(self.m[i])
            simulation.m2f(self.m[i], self.f[i])

            if self.generator.backend.upper() == "LOOPY":
                self.f[i].array_cpu[...] = self.f[i].array.get()

            self.feq[:, self.indices[i]] = self.f[i].swaparray.reshape((nv, self.indices[i].size))
        
        ncond = self.istore.shape[0]
        ind = self.iload[0]
        #m = simulation.container.m.swaparray
        m_11 = simulation.m_halo[2]
        m_s = simulation.m_halo[3]
        m_d = simulation.m_halo[4]
        #bared moments
        m_bar_11 = (1-self.w_11/2)/(1-self.w_11)*m_11[ind[:,1],ind[:,2]]
        m_bar_s = (1-self.w_s/2)/(1-self.w_s)*m_s[ind[:,1],ind[:,2]]
        m_bar_d = (1-self.w_d/2)/(1-self.w_d)*m_d[ind[:,1],ind[:,2]]
        #strain components
        dudx = -.25*(m_bar_s/self.k_nd+m_bar_d/self.mu_nd)
        dvdy = -.25*(m_bar_s/self.k_nd-m_bar_d/self.mu_nd)
        dudydvdx = -1*m_bar_11/self.mu_nd
        
        v = self.stencil.get_all_velocities()
        k = self.iload[0][:,0]
        ksym = self.stencil.get_symmetric()[k]
            
        ind_ax = np.where(ksym<=3)[0]
        ind_diag = np.where(ksym>3)[0]
        self.s_1[ind_ax] = (1-self.theta)*(self.distance[ind_ax]-.5)*(abs(v[ksym[ind_ax]][:,0])*dudx[ind_ax]+abs(v[ksym[ind_ax]][:,1])*dvdy[ind_ax])
        self.s_1[ind_diag] = self.theta/2*(self.distance[ind_diag]-.5)*(dudx[ind_diag]+dvdy[ind_diag]+v[ksym[ind_diag]][:,0]*v[ksym[ind_diag]][:,1]*dudydvdx[ind_diag])
        
    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k][np.newaxis, :]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:] + v[k].T
        self.iload.append(np.concatenate([ksym, indices]))

    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.s_0 + self.s_1
    
    #pylint: disable=too-many-locals
    
    def _get_args(self, ff):
        #need  to delay ff by one time step
        dim = len(ff.nspace)
        nx = ff.nspace[0]
        if dim > 1:
            ny = ff.nspace[1]
        if dim > 2:
            nz = ff.nspace[2]

        
        fcopy = self.f_delay.copy()
        f = ff.array
        self.f_delay = ff.array

        for i in range(len(self.iload)):
            exec('iload{i} = self.iload[{i}]'.format(i=i)) #pylint: disable=exec-used

        istore = self.istore
        rhs = self.rhs
        if hasattr(self, 's'):
            dist = self.s
        ncond = istore.shape[0]
        return locals()
    
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)
        rhs, _ = self._get_rhs_dist_symb(ncond)

        idx = Idx(ix, (0, ncond))
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        fload = indexed('fcopy', [ns, nx, ny, nz], index=[iload[0][idx, k] for k in range(dim+1)], priority=sorder)

        self.generator.add_routine(('dirichlet_le', For(idx, Eq(fstore, fload + rhs[idx]))))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.dirichlet_le

class LENeumann2D(BoundaryMethod):
    """
    Boundary condition of type Neumann for 2D Linear Elasticity

    Notes
    ------

    .. plot:: codes/bounce_back.py

    """
    
    def __init__(self, istore, ilabel, distance, normal, stencil, value_bc, time_bc, nspace, generator):
        super(LENeumann2D, self).__init__(istore, ilabel, distance, normal, stencil, value_bc, time_bc, nspace, generator)
        self.Tx = np.zeros(self.istore.shape[1])
        self.Ty = np.zeros(self.istore.shape[1])
        self.gx = np.zeros(self.istore.shape[1])
        self.gy = np.zeros(self.istore.shape[1])
        self.s_2 = np.zeros(self.istore.shape[1])
        self.s_1 = np.zeros(self.istore.shape[1])
        self.zeta = 2*(np.abs(self.normal[:,1])>np.abs(self.normal[:,0]))-1
        self.a_ijkl = np.zeros((self.istore.shape[1],self.stencil.unvtot))
        self.s_0 = np.zeros(self.istore.shape[1])
        self.iload_neighbor = np.zeros((self.istore.shape[1],self.istore.shape[0]),dtype=int)
        self.has_ax_neighbor = np.zeros(self.istore.shape[1],dtype=bool)
        self.has_diag_neighbor = np.zeros(self.istore.shape[1],dtype=bool)
        self.w_t = 0.
        self.w_11 = 0.
        self.w_s = 0.
        self.w_d = 0.
        self.tau_t = 0.
        self.k_nd = 0.
        self.mu_nd = 0.
        self.theta = 0.

    def prepare_rhs(self, simulation):
        """
        Compute the distribution function at the equilibrium with the value on the border.

        Parameters
        ----------
        simulation : Simulation
            simulation class

        """
        #Initialize Simulation-Parameters for use in update_feq()

        #relaxation rates (w_t = w_12 = w_21)
        self.w_11 = float(simulation.scheme.s.evalf(subs=simulation.scheme.param)[2])
        self.w_s = float(simulation.scheme.s.evalf(subs=simulation.scheme.param)[3])
        self.w_d = float(simulation.scheme.s.evalf(subs=simulation.scheme.param)[4])
        self.w_t = float(simulation.scheme.s.evalf(subs=simulation.scheme.param)[5])
        
        self.tau_t = 1/self.w_t-.5 
        #material properties
        self.k_nd = simulation.scheme.param[symbols('K_ND')]
        self.mu_nd = simulation.scheme.param[symbols('MU_ND')]
        self.theta = simulation.scheme.param[symbols('THETA')]
        
        #compute a_ijkl
        c1 = -(2*(1-self.theta)*(self.k_nd-self.mu_nd))/(self.theta*(1-self.theta-4*self.mu_nd))
        c2 = -2*self.mu_nd/(self.theta-2*self.mu_nd)
        c3 = -4*self.mu_nd/(1-self.theta-4*self.mu_nd)
        
        #number of velocities and symmetric velocities (e.g. kl = [0,1,2,3,4,5,6,7]) 
        v = self.stencil.get_all_velocities()
        vx = v[:,0]
        vy = v[:,1]
        kl = self.stencil.unum - 1
        kl_sym = self.stencil.get_symmetric()[kl]
        k = self.iload[0][0,:]
        ij = self.stencil.get_symmetric()[k]
        #kronecker delta has shape (# of conditions, # of velocities)
        kroneckerdelta = np.atleast_2d(kl_sym) == np.atleast_2d(ij).T

        ind_ax = np.where(ij<=3)[0]
        ind_diag = np.where(ij>3)[0]

        #break down terms a bit and treat ind_ax and ind_diag separately for readability
        
        #-----------axis aligned-----------------------
        t1_kl = np.atleast_2d(np.abs(vx[kl])*np.abs(vy[kl]))
        t1_ij = np.atleast_2d(1+vx[ij[ind_ax]]*self.normal[ind_ax,0]+vy[ij[ind_ax]]*self.normal[ind_ax,1]).T

        t2_kl = np.atleast_2d(vx[kl]*vy[kl])
        t2_ij = np.atleast_2d(vx[ij[ind_ax]]*self.normal[ind_ax,1]+vy[ij[ind_ax]]*self.normal[ind_ax,0]).T
        
        t3_kl_1 = np.atleast_2d(np.abs(vx[kl])*(1-np.abs(vy[kl])))
        t3_kl_2 = np.atleast_2d(np.abs(vy[kl])*(1-np.abs(vx[kl])))
        t3_ij_1 = np.atleast_2d(np.abs(vx[ij[ind_ax]])+vx[ij[ind_ax]]*self.normal[ind_ax,0]).T
        t3_ij_2 = np.atleast_2d(np.abs(vy[ij[ind_ax]])+vy[ij[ind_ax]]*self.normal[ind_ax,1]).T
        
        self.a_ijkl[ind_ax,:] = c1*t1_kl*t1_ij+c2*t2_kl*t2_ij+c3*(t3_kl_1*t3_ij_1+t3_kl_2*t3_ij_2)-kroneckerdelta[ind_ax,:]
        
        #-------------diagonal--------------------------
        #kl terms are the same
        t1_ij = np.atleast_2d((.5*(1+self.zeta[ind_diag]))*vx[ij[ind_diag]]*self.normal[ind_diag,0]+(.5*(1-self.zeta[ind_diag]))*vy[ij[ind_diag]]*self.normal[ind_diag,1]).T

        t2_ij = np.atleast_2d(vx[ij[ind_diag]]*vy[ij[ind_diag]]+(.5*(1+self.zeta[ind_diag]))*vx[ij[ind_diag]]*self.normal[ind_diag,1]+(.5*(1-self.zeta[ind_diag]))*vy[ij[ind_diag]]*self.normal[ind_diag,0]).T

        t3_ij_1 = np.atleast_2d((.5*(1+self.zeta[ind_diag]))*vx[ij[ind_diag]]*self.normal[ind_diag,0]).T
        t3_ij_2 = np.atleast_2d((.5*(1-self.zeta[ind_diag]))*vy[ij[ind_diag]]*self.normal[ind_diag,1]).T
        
        self.a_ijkl[ind_diag,:] = .5*c1*t1_kl*t1_ij+.5*c2*t2_kl*t2_ij+0.5*c3*(t3_kl_1*t3_ij_1+t3_kl_2*t3_ij_2)-kroneckerdelta[ind_diag,:]
        
        #neighbor search
        #search only necessary for bc's with diagonal vel.
        
        has_ax_neighbor = np.zeros(ind_diag.shape[0],dtype=bool)
        has_diag_neighbor = np.zeros(ind_diag.shape[0],dtype=bool)
        ij_hat_ax_neighbor = np.zeros(ind_diag.shape[0],dtype=int)
        ij_hat_diag_neighbor = np.zeros(ind_diag.shape[0],dtype=int)
        x_i_neighbor = np.zeros(ind_diag.shape[0],dtype=int)
        y_i_neighbor = np.zeros(ind_diag.shape[0],dtype=int)

        #home indices 
        x_i_h = self.iload[0][1,ind_diag]
        y_i_h = self.iload[0][2,ind_diag]
        zeta_h = self.zeta[ind_diag]
        
        #indices where boundary is more horizontal or more vertical    
        ihori = np.where(zeta_h == 1)[0]
        ivert = np.where(zeta_h == -1)[0]
        
        #search in ij direction from home, but store ij_hat-direction
        v_search = self.stencil.get_all_velocities()
        ij_hat = self.stencil.get_symmetric()[ij]
        
        #more close to horizontal boundary
        
        for ij in [1,3]:
            ij_hat = self.stencil.get_symmetric()[ij]
            search = simulation.domain.in_or_out[x_i_h[ihori]+v_search[ij][0],y_i_h[ihori]+v_search[ij][1]] == simulation.domain.valin
            has_ax_neighbor[ihori] = has_ax_neighbor[ihori] | search
            idx_update_direction = np.where(search)[0]
            temp = ij_hat_ax_neighbor[ihori]
            temp[idx_update_direction] = ij_hat*search[idx_update_direction] 
            ij_hat_ax_neighbor[ihori] = temp 
            
        for ij in [4,5,6,7]:
            ij_hat = self.stencil.get_symmetric()[ij]
            search = simulation.domain.in_or_out[x_i_h[ihori]+v_search[ij][0],y_i_h[ihori]+v_search[ij][1]] == simulation.domain.valin
            has_diag_neighbor[ihori] = has_diag_neighbor[ihori] | search
            idx_update_direction = np.where(search)[0]
            temp = ij_hat_ax_neighbor[ihori]
            temp[idx_update_direction] = ij_hat*search[idx_update_direction] 
            ij_hat_diag_neighbor[ihori] = temp 
        
        #more close to vertical boundary
        
        for ij in [0,2]:
            ij_hat = self.stencil.get_symmetric()[ij]
            search = simulation.domain.in_or_out[x_i_h[ivert]+v_search[ij][0],y_i_h[ivert]+v_search[ij][1]] == simulation.domain.valin
            has_ax_neighbor[ivert] = has_ax_neighbor[ivert] | search
            idx_update_direction = np.where(search)[0]
            temp = ij_hat_ax_neighbor[ivert]
            temp[idx_update_direction] = ij_hat*search[idx_update_direction] 
            ij_hat_ax_neighbor[ivert] = temp 

        for ij in [4,5,6,7]:
            ij_hat = self.stencil.get_symmetric()[ij]
            search = simulation.domain.in_or_out[x_i_h[ivert]+v_search[ij][0],y_i_h[ivert]+v_search[ij][1]] == simulation.domain.valin
            has_diag_neighbor[ivert] = has_diag_neighbor[ivert] | search
            idx_update_direction = np.where(search)
            temp = ij_hat_diag_neighbor[ivert]
            temp[idx_update_direction] = ij_hat*search[idx_update_direction] 
            ij_hat_diag_neighbor[ivert] = temp 
        
        #make sure we only use diag neighbor if no ax neighbor is available
        has_diag_neighbor = has_diag_neighbor & np.logical_not(has_ax_neighbor)
        has_any_neighbor = has_diag_neighbor | has_ax_neighbor
        has_no_neighbor = np.logical_not(has_any_neighbor)
        
        #ij_hat for all_bc's; ij_hat is set to -1 where no neighbour is available
        ij_hat_neighbor = ij_hat_ax_neighbor*has_ax_neighbor + ij_hat_diag_neighbor*has_diag_neighbor + (-1*has_no_neighbor)
        
        #compute neighbor x and y index
        x_i_neighbor[has_any_neighbor] = x_i_h[has_any_neighbor] - v_search[ij_hat_neighbor[has_any_neighbor]][:,0]
        y_i_neighbor[has_any_neighbor] = y_i_h[has_any_neighbor] - v_search[ij_hat_neighbor[has_any_neighbor]][:,1]
        
        #store neighbor search in self.iload_neighbor: only bc's whith missing diagonal velocities have nonzero entries
        #first column has ij_hat of neighbor node (-1 if no neighbor is available)
        #second and third column the x and y index of that neighbor node
        self.iload_neighbor[ind_diag,0] = ij_hat_neighbor
        self.iload_neighbor[ind_diag,1] = x_i_neighbor
        self.iload_neighbor[ind_diag,2] = y_i_neighbor
        self.has_ax_neighbor[ind_diag] = has_ax_neighbor
        self.has_diag_neighbor[ind_diag] = has_diag_neighbor

        nv = simulation.container.nv
        sorder = simulation.container.sorder
        nspace = [1]*(len(sorder)-1)
        v = self.stencil.get_all_velocities()

        gpu_support = simulation.container.gpu_support
        
        #Get forcing as function handle
        consm = []
        consmi = []
        for k_,v_ in simulation.scheme.consm.items():
            consm.append(k_)
            consmi.append(v_)
        
        #2D coordinates [X,Y]
        symb_coord = simulation.scheme.symb_coord.copy()
        symb_coord.pop()
        #symbolic forcing terms and function handles
        if simulation.scheme._source_terms[0] is not None: 
            if isinstance(simulation.scheme._source_terms[0][consm[0]], (float, int)):
                def g_x_f(x,y): return simulation.scheme._source_terms[0][consm[0]]*np.ones(x.shape)
            else:
                g_x_sym = simulation.scheme._source_terms[0][consm[0]]
                import sympy as sp
                g_x_f = sp.lambdify(symb_coord,g_x_sym.evalf(subs=simulation.scheme.param))
            if isinstance(simulation.scheme._source_terms[0][consm[1]], (float, int)):
                def g_y_f(x,y): return simulation.scheme._source_terms[0][consm[1]]*np.ones(x.shape)
            else:
                g_y_sym = simulation.scheme._source_terms[0][consm[1]]
                import sympy as sp
                g_y_f = sp.lambdify(symb_coord,g_y_sym.evalf(subs=simulation.scheme.param))
        else:
            #dummy functions if no source term is given 
            def g_x_f(x,y): return 0.*x*y 
            def g_y_f(x,y): return 0.*x*y

        for key, value in self.value_bc.items():
            if value is not None:
                indices = np.where(self.ilabel == key)
                # TODO: check the index in sorder to be the most contiguous
                nspace[0] = indices[0].size
                k = self.istore[0, indices]

                s = 1 - self.distance[indices]
                coords = tuple()
                for i in range(simulation.domain.dim):
                    x = simulation.domain.coords_halo[i][self.istore[i + 1, indices]]
                    x += s*v[k, i]*simulation.domain.dx
                    x = x.ravel()
                    for j in range(1, simulation.domain.dim): #pylint: disable=unused-variable
                        x = x[:, np.newaxis]
                    coords += (x,)

                m = Array(nv, nspace, 0, sorder, gpu_support=gpu_support)
                m.set_conserved_moments(simulation.scheme.consm)

                f = Array(nv, nspace, 0, sorder, gpu_support=gpu_support)
                f.set_conserved_moments(simulation.scheme.consm)

                args = coords
                if isinstance(value, types.FunctionType):
                    func = value
                elif isinstance(value, tuple):
                    func = value[0]
                    args += value[1]

                if self.time_bc[key]:
                    func(f, m, 0, *args)
                else:
                    func(f, m, *args)
             

                k = self.iload[0][0,:]
                ksym = self.stencil.get_symmetric()[k]
                #xy-coordinates of boundary nodes
                x_g = simulation.domain.x_halo[self.iload[0][1,indices[0]]]
                y_g = simulation.domain.y_halo[self.iload[0][2,indices[0]]]
                
                g_x = g_x_f(x_g,y_g)
                g_y = g_y_f(x_g,y_g)
                m_index = 0
                for icond in indices[0]:
                    v_i = ksym[icond]
                    self.Tx[icond] = m[0,m_index]
                    self.Ty[icond] = m[1,m_index]
                    self.gx[icond] = g_x[m_index]
                    self.gy[icond] = g_y[m_index]
                    #s_1 (general)
                    if v_i <= 3:
                        self.s_1[icond] = v[v_i][0]*self.Tx[icond]+ v[v_i][1]*self.Ty[icond]
                    else:
                        self.s_1[icond] = .25*(v[v_i][0]*(1+self.zeta[icond])*self.Tx[icond]+v[v_i][1]*(1-self.zeta[icond])*self.Ty[icond])
                    m_index+=1            

                if self.generator.backend.upper() == "LOOPY":
                    f.array_cpu[...] = f.array.get()

                self.feq[:, indices[0]] = f.swaparray.reshape((nv, indices[0].size))

                if self.time_bc[key]:
                    self.func.append(func)
                    self.args.append(args)
                    self.f.append(f)
                    self.m.append(m)
                    self.indices.append(indices[0])

    def update_feq(self,simulation):
        """
        override this method to access simulation.m 
        """
        t = simulation.t
        nv = simulation.container.nv

        for i in range(len(self.func)):
            self.func[i](self.f[i], self.m[i], t, *self.args[i])
            simulation.equilibrium(self.m[i])
            simulation.m2f(self.m[i], self.f[i])

            if self.generator.backend.upper() == "LOOPY":
                self.f[i].array_cpu[...] = self.f[i].array.get()

            self.feq[:, self.indices[i]] = self.f[i].swaparray.reshape((nv, self.indices[i].size))
        
        v = self.stencil.get_all_velocities()
        k = self.iload[0][:,0]
        ksym = self.stencil.get_symmetric()[k]
        ind_ax = np.where(ksym<=3)[0]
        mask_diag = ksym>3
        ind_diag = np.where(ksym>3)[0]
        ncond = self.istore.shape[0]
        ind = self.iload[0]
        ind_n = self.iload_neighbor
        mask_hori = (self.zeta == 1)
        mask_vert = (self.zeta == -1)
        #Post-collision moments 
        m_10 = simulation.m_halo[0]
        m_01 = simulation.m_halo[1]
        m_11 = simulation.m_halo[2]
        m_s = simulation.m_halo[3]
        m_d = simulation.m_halo[4]
        m_12 = simulation.m_halo[5]
        m_21 = simulation.m_halo[6]
        #bared and precollision moments
        m_bar_10 = m_10[ind[:,1],ind[:,2]]-.5*self.gx
        m_bar_01 = m_01[ind[:,1],ind[:,2]]-.5*self.gy
        m_bar_11 = (1-self.w_11/2)/(1-self.w_11)*m_11[ind_n[:,1],ind_n[:,2]]
        m_bar_s = (1-self.w_s/2)/(1-self.w_s)*m_s[ind[:,1],ind[:,2]]
        m_bar_s_n = (1-self.w_s/2)/(1-self.w_s)*m_s[ind_n[:,1],ind_n[:,2]]
        m_bar_d = (1-self.w_d/2)/(1-self.w_d)*m_d[ind[:,1],ind[:,2]]
        m_bar_d_n = (1-self.w_d/2)/(1-self.w_d)*m_d[ind_n[:,1],ind_n[:,2]]
        m_pre_12 = (m_12[ind[:,1],ind[:,2]]-self.w_t*self.theta*m_bar_10)/(1-self.w_t)
        m_pre_21 = (m_21[ind[:,1],ind[:,2]]-self.w_t*self.theta*m_bar_01)/(1-self.w_t)
        #stress components from moments
        sigxx = -0.5*(m_bar_s+m_bar_d)[mask_diag&mask_hori]
        sigxx_n = -0.5*(m_bar_s_n+m_bar_d_n)[mask_diag&mask_hori]
        sigyy = -0.5*(m_bar_s-m_bar_d)[mask_diag&mask_vert]
        sigyy_n = -0.5*(m_bar_s_n-m_bar_d_n)[mask_diag&mask_vert]
        #stress derivatives from moments
        dxsigxx = 2/(1+2*self.tau_t)*(self.theta*m_bar_10-m_pre_12)-self.gx
        dysigyy = 2/(1+2*self.tau_t)*(self.theta*m_bar_01-m_pre_21)-self.gy
        dysigxy = 2/(1+2*self.tau_t)*(m_pre_12-self.theta*m_bar_10)
        dxsigxy = 2/(1+2*self.tau_t)*(m_pre_21-self.theta*m_bar_01)
        dysigxx = v[ind_n[:,0][mask_diag&mask_hori]][:,1]*(sigxx-sigxx_n)
        dysigxx[self.has_diag_neighbor[mask_diag&mask_hori]] -= v[ind_n[:,0][mask_diag&mask_hori&self.has_diag_neighbor]][:,1]*v[ind_n[:,0][mask_diag&mask_hori&self.has_diag_neighbor]][:,0]*dxsigxx[mask_diag&mask_hori&self.has_diag_neighbor]
        dxsigyy = v[ind_n[:,0][mask_diag&mask_vert]][:,0]*(sigyy-sigyy_n)
        dxsigyy[self.has_diag_neighbor[mask_diag&mask_vert]] -= v[ind_n[:,0][mask_diag&mask_vert&self.has_diag_neighbor]][:,1]*v[ind_n[:,0][mask_diag&mask_vert&self.has_diag_neighbor]][:,0]*dysigyy[mask_diag&mask_vert&self.has_diag_neighbor]
        
        #source term for axis aligned velocities        
        self.s_2[ind_ax] = .5*(v[ksym[ind_ax]][:,0]*dxsigxx[ind_ax]+v[ksym[ind_ax]][:,1]*dysigyy[ind_ax])+self.distance[ind_ax]*(np.abs(v[ksym[ind_ax]][:,0])*(dxsigxx[ind_ax]*self.normal[ind_ax,0]+dxsigxy[ind_ax]*self.normal[ind_ax,1])+np.abs(v[ksym[ind_ax]][:,1])*(dysigxy[ind_ax]*self.normal[ind_ax,0]+dysigyy[ind_ax]*self.normal[ind_ax,1]))
        
        #source term for diagonal velocities
        self.s_2[mask_diag] =  .25*(v[ksym[ind_diag]][:,0]*dysigxy[ind_diag]+v[ksym[ind_diag]][:,1]*dxsigxy[ind_diag])
        
        #correction for horizontal neighbors
        self.s_2[mask_diag&mask_hori] += .5*self.distance[mask_diag&mask_hori]*(dxsigxx[mask_diag&mask_hori]*self.normal[mask_diag&mask_hori,0]+dxsigxy[mask_diag&mask_hori]*self.normal[mask_diag&mask_hori,1]+v[ksym[mask_diag&mask_hori]][:,0]*v[ksym[mask_diag&mask_hori]][:,1]*(dysigxx*self.normal[mask_diag&mask_hori,0]+dysigxy[mask_diag&mask_hori]*self.normal[mask_diag&mask_hori,1]))
        
        #correction for vertical neighbors
        self.s_2[mask_diag&mask_vert] += .5*self.distance[mask_diag&mask_vert]*(dysigxy[mask_diag&mask_vert]*self.normal[mask_diag&mask_vert,0]+dysigyy[mask_diag&mask_vert]*self.normal[mask_diag&mask_vert,1]+v[ksym[mask_diag&mask_vert]][:,0]*v[ksym[mask_diag&mask_vert]][:,1]*(dxsigxy[mask_diag&mask_vert]*self.normal[mask_diag&mask_vert,0]+dxsigyy*self.normal[mask_diag&mask_vert,1]))
        
        #has no neighbor -> no 2nd order correction possible
        self.s_2[ind_n[:,0]==-1] = 0.
        
        
    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k][np.newaxis, :]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:] + v[k].T
        self.iload.append(np.concatenate([ksym, indices]))

    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.s_1 + self.s_2
    
    #pylint: disable=too-many-locals
    
    def _get_args(self, ff):
        #need  to delay ff by one time step
        dim = len(ff.nspace)
        nx = ff.nspace[0]
        if dim > 1:
            ny = ff.nspace[1]
        if dim > 2:
            nz = ff.nspace[2]

        f = ff.array

        for i in range(len(self.iload)):
            exec('iload{i} = self.iload[{i}]'.format(i=i)) #pylint: disable=exec-used

        istore = self.istore
        rhs = self.rhs
        a_ijkl = self.a_ijkl
        if hasattr(self, 's'):
            dist = self.s
        ncond = istore.shape[0]
        return locals()
    
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix
        from sympy import Add

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)
        rhs, _ = self._get_rhs_dist_symb(ncond)
        a_ijkl = IndexedBase('a_ijkl', [ncond,self.stencil.unvtot])

        idx = Idx(ix, (0, ncond))
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        
        fload = []
        for ij in range(ns):
            fload.append(indexed('f', [ns, nx, ny, nz], index=[ij,iload[0][idx, 1], iload[0][idx, 2]], priority=sorder))
        
        self.generator.add_routine(('neumann_le', For(idx, Eq(fstore, Add(*[a_ijkl[idx,ij]*fload[ij] for ij in range(ns)]) + rhs[idx])))) 

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.neumann_le

class BouzidiBounceBack(BoundaryMethod):
    """
    Boundary condition of type Bouzidi bounce-back [BFL01]

    Notes
    ------

    .. plot:: codes/Bouzidi.py

    """
    def __init__(self, istore, ilabel, distance, normal, stencil, value_bc, time_bc, nspace, generator):
        super(BouzidiBounceBack, self).__init__(istore, ilabel, distance, normal, stencil, value_bc, time_bc, nspace, generator)
        self.s = np.empty(self.istore.shape[1])

    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        ksym = self.stencil.get_symmetric()[k]
        v = self.stencil.get_all_velocities()

        iload1 = np.zeros(self.istore.shape, dtype=np.int32)
        iload2 = np.zeros(self.istore.shape, dtype=np.int32)

        mask = self.distance < .5
        iload1[0, mask] = ksym[mask]
        iload2[0, mask] = ksym[mask]
        iload1[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        iload2[1:, mask] = self.istore[1:, mask] + 2*v[k[mask]].T
        self.s[mask] = 2.*self.distance[mask]

        mask = np.logical_not(mask)
        iload1[0, mask] = ksym[mask]
        iload2[0, mask] = k[mask]
        iload1[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        iload2[1:, mask] = self.istore[1:, mask] + v[k[mask]].T
        self.s[mask] = .5/self.distance[mask]

        self.iload.append(iload1)
        self.iload.append(iload2)

    def _get_args(self, ff):
        dim = len(ff.nspace)
        nx = ff.nspace[0]
        if dim > 1:
            ny = ff.nspace[1]
        if dim > 2:
            nz = ff.nspace[2]

        f = ff.array
        # FIXME: needed to have the same results between numpy and cython
        # That means that there are dependencies between the rhs and the lhs
        # during the loop over the boundary elements
        # check why (to test it use air_conditioning example)
        fcopy = ff.array.copy()

        for i in range(len(self.iload)):
            exec('iload{i} = self.iload[{i}]'.format(i=i)) #pylint: disable=exec-used

        istore = self.istore
        rhs = self.rhs
        if hasattr(self, 's'):
            dist = self.s
        ncond = istore.shape[0]
        return locals()

    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.feq[k, np.arange(k.size)] - self.feq[ksym, np.arange(k.size)]

    #pylint: disable=too-many-locals
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)
        rhs, dist = self._get_rhs_dist_symb(ncond)

        idx = Idx(ix, (0, ncond))
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        fload0 = indexed('fcopy', [ns, nx, ny, nz], index=[iload[0][idx, k] for k in range(dim+1)], priority=sorder)
        fload1 = indexed('fcopy', [ns, nx, ny, nz], index=[iload[1][idx, k] for k in range(dim+1)], priority=sorder)

        self.generator.add_routine(('Bouzidi_bounce_back', For(idx, Eq(fstore, dist[idx]*fload0 + (1-dist[idx])*fload1 + rhs[idx]))))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.Bouzidi_bounce_back

class AntiBounceBack(BounceBack):
    """
    Boundary condition of type anti bounce-back

    Notes
    ------

    .. plot:: codes/anti_bounce_back.py

    """
    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.feq[k, np.arange(k.size)] + self.feq[ksym, np.arange(k.size)]

    #pylint: disable=too-many-locals
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)
        rhs, _ = self._get_rhs_dist_symb(ncond)

        idx = Idx(ix, (0, ncond))
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        fload = indexed('f', [ns, nx, ny, nz], index=[iload[0][idx, k] for k in range(dim+1)], priority=sorder)

        self.generator.add_routine(('anti_bounce_back', For(idx, Eq(fstore, -fload + rhs[idx]))))

    @property
    def function(self):
        return self.generator.module.anti_bounce_back

class BouzidiAntiBounceBack(BouzidiBounceBack):
    """
    Boundary condition of type Bouzidi anti bounce-back

    Notes
    ------

    .. plot:: codes/Bouzidi.py

    """
    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        k = self.istore[:, 0]
        ksym = self.stencil.get_symmetric()[k]
        self.rhs[:] = self.feq[k, np.arange(k.size)] + self.feq[ksym, np.arange(k.size)]

    #pylint: disable=too-many-locals
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)
        rhs, dist = self._get_rhs_dist_symb(ncond)

        idx = Idx(ix, (0, ncond))
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        fload0 = indexed('f', [ns, nx, ny, nz], index=[iload[0][idx, k] for k in range(dim+1)], priority=sorder)
        fload1 = indexed('f', [ns, nx, ny, nz], index=[iload[1][idx, k] for k in range(dim+1)], priority=sorder)

        self.generator.add_routine(('Bouzidi_anti_bounce_back', For(idx, Eq(fstore, -dist[idx]*fload0 + (1-dist[idx])*fload1 + rhs[idx]))))

    @property
    def function(self):
        return self.generator.module.Bouzidi_anti_bounce_back

class Neumann(BoundaryMethod):
    """
    Boundary condition of type Neumann

    """
    name = 'neumann'
    def set_rhs(self):
        """
        Compute and set the additional terms to fix the boundary values.
        """
        pass

    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:] + v[k].T
        self.iload.append(np.concatenate([k[np.newaxis, :], indices]))

    #pylint: disable=too-many-locals
    def generate(self, sorder):
        """
        Generate the numerical code.

        Parameters
        ----------
        sorder : list
            the order of nv, nx, ny and nz
        """
        from .generator import For
        from .symbolic import nx, ny, nz, indexed, ix

        ns = int(self.stencil.nv_ptr[-1])
        dim = self.stencil.dim

        istore, iload, ncond = self._get_istore_iload_symb(dim)

        idx = Idx(ix, (0, ncond))
        fstore = indexed('f', [ns, nx, ny, nz], index=[istore[idx, k] for k in range(dim+1)], priority=sorder)
        fload = indexed('f', [ns, nx, ny, nz], index=[iload[0][idx, k] for k in range(dim+1)], priority=sorder)

        self.generator.add_routine((self.name, For(idx, Eq(fstore, fload))))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.neumann

class NeumannX(Neumann):
    """
    Boundary condition of type Neumann along the x direction

    """
    name = 'neumannx'
    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:].copy()
        indices[0] += v[k].T[0]
        self.iload.append(np.concatenate([k[np.newaxis, :], indices]))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.neumannx

class NeumannY(Neumann):
    """
    Boundary condition of type Neumann along the y direction

    """
    name = 'neumanny'
    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:].copy()
        indices[1] += v[k].T[1]
        self.iload.append(np.concatenate([k[np.newaxis, :], indices]))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.neumanny

class NeumannZ(Neumann):
    """
    Boundary condition of type Neumann along the z direction

    """
    name = 'neumannz'
    def set_iload(self):
        """
        Compute the indices that are needed (symmertic velocities and space indices).
        """
        k = self.istore[0]
        v = self.stencil.get_all_velocities()
        indices = self.istore[1:].copy()
        indices[1] += v[k].T[2]
        self.iload.append(np.concatenate([k[np.newaxis, :], indices]))

    @property
    def function(self):
        """Return the generated function"""
        return self.generator.module.neumannz
