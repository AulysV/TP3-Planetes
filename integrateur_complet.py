# -*- coding: utf-8 -*-
"""
@author: Y. Vadée Le Brun

DESCRIPTION

Ce code propose plusieurs solveurs d'équations différentielles
"""

"""
BIBLIOTHEQUES
"""
# import de la bibliothèque numpy (gestion de matrices et routines mathématiques) en lui donnant le surnom np
import numpy as np
import time
from functools import lru_cache

class ODESolver:
    def __init__(self, f):
        self.model = f
        self.f = f.derA
        
    def solve(self, u0, temps, dt):
        self.dt = dt
        # Initialisation de la CI
        if np.isscalar(u0): # ODE scalaire
            u0 = float(u0)
            self.neq = 1
        else: # ODE vectorielle
            u0 = np.asarray(u0)
            self.neq = u0.size
        self.u0 = u0

        # self.u est le tableau qui contiendra la solution calculée à tous les instants de temps
        N = temps.size
        if self.neq == 1:
            self.u = np.zeros(N)
        else:
            self.u = np.zeros((N, self.neq))
        self.u[0] = self.u0

        # self.t stocke l'instant t de la résolution
        self.t = temps[0]
        # self.ut stocke la solution u à l'instant t
        self.ut = self.u0

        # Boucle de calcul qui se charge de remplir le tableau u
        for n in range(1, N):
            # Il faut calculer combien de pas de temps sont nécessaire pour arriver à la prochaine case
            ti = temps[n-1]
            tf = temps[n]
            a = (tf - ti) / dt
            nb_steps = max(int(np.round(a)),1)
            tempdt = (tf - ti)/nb_steps
            for i in range(nb_steps):
                self.advance(tempdt)
                self.t = ti + (i+1)*tempdt
                #print("Calcul à t =", self.t)

            t_rest = tf - self.t
            assert t_rest > -dt, "Error in time calculation t_rest should be positive"
            assert t_rest < dt, "Error in time calculation t_rest should be smaller than dt"   
            self.u[n] = self.ut

        return self.u

    def advance(self, dt):
        raise NotImplementedError("Advance method is not implemented in the base class")
    
    def return_error(self, temps, dt, u_ref):
        model = self.model
        if np.isscalar(dt):
            print(type(self).__name__, ": Calcul de l'erreur pour", dt)
            start = time.perf_counter()
            u = self.solve(model.CI(), temps, dt)
            end = time.perf_counter()
            elapsed = (end - start) * 1000
            print("Temps d'éxécution :", elapsed)
            error_t = np.abs(u[:,0]-u_ref[:,0])
            # L'erreur maximale est le maximum de ce tableau
            error = np.max(error_t)
            print("erreur :", error)
            return elapsed, error
        
        else:
            elapsed_list=[]
            error_list=[]
            for dtp in dt:
                elapsed, error = self.return_error(temps, dtp, u_ref)
                elapsed_list.append(elapsed)
                error_list.append(error)
            return elapsed_list, error_list
    
class ForwardEuler(ODESolver):
    def advance(self, dt):
        """Advance the solution one time step."""
        u, f, t = self.ut, self.f, self.t
        u += f(t,u) * dt
    
class ExplicitMidpoint(ODESolver):
    def advance(self, dt):
        u, f, t = self.ut, self.f, self.t
        dt2 = dt / 2.0
        k1 = f(t, u)
        k2 = f(t + dt2, u + dt2 * k1)
        self.ut += dt * k2
    
class RungeKutta4(ODESolver):
    def advance(self, dt):
        u, f, t = self.ut, self.f, self.t
        dt2 = dt / 2.0
        k1 = f(t, u,)
        k2 = f(t + dt2, u + dt2 * k1, )
        k3 = f(t + dt2, u + dt2 * k2, )
        k4 = f(t + dt, u + dt * k3, )
        self.ut += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)