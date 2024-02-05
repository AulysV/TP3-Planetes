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

class MecaODESolver:
    def __init__(self, f):
        self.model = f
        self.f = f.acc
        
    def solve(self, u0, temps, dt):
        self.dt = dt
        # Initialisation de la CI
        u0 = np.asarray(u0)
        self.neq = int(u0.size/2)
        self.pos0, self.vel0 = np.split(u0,2)

        # self.u est le tableau qui contiendra la solution calculée à tous les instants de temps
        N = temps.size
        self.pos = np.zeros((N, self.neq))
        self.vel = np.zeros((N, self.neq))
        self.pos[0] = self.pos0
        self.vel[0] = self.vel0

        # self.t stocke l'instant t de la résolution
        self.t = temps[0]
        # self.ut stocke la solution u à l'instant t
        self.post = self.pos0
        self.velt = self.vel0

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

            self.pos[n] = self.post
            self.vel[n] = self.velt
                
        self.u = np.concatenate((self.pos, self.vel), axis = 1)

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
    
# La version entièrement vectorisée dans ODESOlver est plus rapide
# Cette version n'a donc aucun intérêt.    
class MecaForwardEuler(MecaODESolver):
    def advance(self, dt):
        """Advance the solution one time step."""
        f, t, pos, vel = self.f, self.t, self.post, self.velt
        k = f(t, pos, vel)
        pos += vel * dt
        vel += k * dt
        

class MecaVelocityVerlet(MecaODESolver):
    def advance(self, dt):
        
        f, t, pos, vel = self.f, self.t, self.post, self.velt
        dt2 = dt / 2.0
        
        vel += f(t, pos, vel) * dt2
        pos += vel * dt
        vel += f(t + dt, pos, vel) * dt2

# C'est la méthode de Verlet mais dans le cas où on n'a pas besoin de la vitesse à chaque pas de temps
# Avantage : plus rapide       
class Stormer_Verlet(MecaODESolver):
    def solve(self, u0, temps, dt):
        self.dt = dt
        # Initialisation de la CI
        if np.isscalar(u0): # ODE scalaire
            u0 = float(u0)
            self.neq = 1
        else: # ODE vectorielle
            u0 = np.asarray(u0)
            self.neq = int(u0.size/2)
        self.pos0, self.vel0 = np.split(u0,2)

        # self.u est le tableau qui contiendra la solution calculée à tous les instants de temps
        N = temps.size
        if self.neq == 1:
            self.pos = np.zeros(N)
            self.vel = np.zeros(N)
        else:
            self.pos = np.zeros((N, self.neq))
            self.vel = np.zeros((N, self.neq))
        self.pos[0] = self.pos0
        self.vel[0] = self.vel0

        # self.t stocke l'instant t de la résolution
        self.t = temps[0]
        # self.ut stocke la solution u à l'instant t
        self.post = self.pos0
        self.oldpost = np.copy(self.pos0)
        self.velt = self.vel0

        # Boucle de calcul qui se charge de remplir le tableau des positions
        # La première fois est spéciale
        ti = temps[0]
        tf = temps[1]
        a = (tf - ti) / dt
        nb_steps = max(int(np.round(a)),1)
        tempdt = (tf - ti)/nb_steps
        self.post += self.velt * tempdt + 1/2*self.f(self.t, self.post, self.velt) * tempdt**2
        self.t = ti + tempdt
        for i in range(1, nb_steps):
            self.advance(tempdt)
            self.t = ti + (i+1)*tempdt
            #print("Calcul à t =", self.t)
        self.pos[1] = self.post
        futur_pos = 2 * self.post - self.oldpost + self.f(self.t, self.post, self.velt) * tempdt**2
        self.vel[1] = (futur_pos - self.oldpost) / (2 * tempdt)

        for n in range(2, N):
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
            self.pos[n] = self.post
            futur_pos = 2 * self.post - self.oldpost + self.f(self.t, self.post, self.velt) * tempdt**2
            self.vel[n] = (futur_pos - self.oldpost) / (2 * tempdt)

        """
        # Il faut rajouter la vitesse qui n'est pas calculée de base
        for n in range(1, N-1):
            self.vel[n] = (self.pos[n+1] - self.pos[n-1]) / (temps[n+1] - temps[n-1])
        self.vel[N-1] =  (self.pos[N-1] - self.pos[N-2]) / (temps[N-1] - temps[N-2])              
        """
        self.u = np.hstack((self.pos, self.vel))
        return self.u
    def advance(self, dt):
        
        f, t, pos, vel = self.f, self.t, self.post, self.velt

        k = f(t, pos, vel)
        temp = np.copy(pos)
        pos += pos - self.oldpost + k * dt**2
        self.oldpost = temp
