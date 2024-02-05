# -*- coding: utf-8 -*-
"""
@author: Y. Vadée Le Brun

DESCRIPTION

Ce code stocke la classe planète.
"""

"""
BIBLIOTHEQUES
"""
# import de la bibliothèque numpy (gestion de matrices et routines mathématiques) en lui donnant le surnom np
import numpy as np

"""
CLASSE PLANETE
"""

class Planet:
    # Le constructeur de la planete. Notez que les paramètres ont une valeur par défaut.
    # Cela permet de ne pas tout spécifier à chaque fois.
    # Les paramètres par défauts correspondent à la Terre, en supposant son orbite circulaire
    def __init__(self, theta_0 = 0, r_0 = 149.5*10**9, omega_0 = 2.0*np.pi/(365*3600*24), rpoint_0 = 0, cyl = False):

        # Constante de gravitation universelle
        self.G = 6.67*10**-11
        #Masse du soleil
        self.M = 1.98892*10**30
        self.k = self.G*self.M

        #Paramètres initiaux
        # Rayon de l'orbite terrestre
        self.theta_0 = theta_0
        self.r_0 = r_0
        self.omega_0 = omega_0
        self.rpoint_0 = rpoint_0

        # Calculs de paramètres de l'ellipse :
        self.C = r_0**2 * omega_0
        self.p = self.C**2 / self.k
        self.theta_p = self.theta_0 + np.arctan((self.p * r_0 * rpoint_0)/(self.C*(r_0 - self.p)))
        #self.e = (self.C**2 / (self.k * r_0) - 1)
        self.e = ((self.p / r_0) - 1) * 1/np.cos(self.theta_0 - self.theta_p)

        # Un booléen pour savoir si le mouvement est étudié en cylindriques. Sinon en cartésiennes.
        self.cyl = cyl

    def CI(self):
        # On créé un unique vecteur qui contient position et vitesse pour des raisons algorithmiques
        # Le vecteur A est un vecteur colonne de 4 cases : les deux premières pour theta et r.
        # les deux suivantes pour la vitesse angulaire et la vitesse radiale.
        if self.cyl:
            return np.array((self.theta_0, self.r_0, self.omega_0, self.rpoint_0))
        # En coordonnées cartésiennes, A contient x,y puis vx, vy.
        # L'axe des x pointe vers la position initiale de la planète
        else:
            r_0, rpoint_0, theta_0, omega_0 = self.r_0, self.rpoint_0, self.theta_0, self.omega_0
            x = r_0 * np.cos(theta_0)
            y = r_0 * np.sin(theta_0)
            vx = rpoint_0 * np.cos(theta_0) - r_0 * omega_0 * np.sin(theta_0)
            vy = rpoint_0 * np.sin(theta_0) + r_0 * omega_0 * np.cos(theta_0)
            return np.array((x,y, vx, vy))

    # Cette fonction correspond au G du polycopié. Elle renvoie la dérivée du vecteur A.
    def derA(self, t, A):
        if self.cyl:
            return self.der_A_cyl(t, A)
        else:
            return self.der_A_cart(t, A)
        
    def acc(self, t, pos, vel):
        if self.cyl:
            return self.acc_cyl(t, pos, vel)
        else:
            return self.acc_cart(t, pos, vel)
        
        
    def Em(self, A):
        if self.cyl:
            return self.Em_cyl(A)
        else:
            return self.Em_cart(A)
        
    def L(self, A):
        if self.cyl:
            return self.L_cyl(A)
        else:
            return self.L_cart(A)
        
    # Cylindriques

    def der_A_cyl(self, t, A) :
        # la fonctions empty de numpy créé un tableau non initialisé de n case
        B = np.empty(4)
        # TODO: Expressions pour les dérivées par rapport au temps en coordonnées cylindriques
        B[0] = A[2]
        B[1] = A[3]
        B[2] = self.acc_cyl(t, A[0], A[1])
        B[3] = self.der_r(A[0]) * A[2]**2 + self.r(A[0]) * self.acc_cyl(t, A[0], A[1])
        return B

    def acc_cyl(self, t, pos, vel) :
        # la fonctions empty de numpy créé un tableau non initialisé de n case
        B = np.empty(2)
        # TODO: Expression pour l'accélération radiale en coordonnées cylindriques
        B[0] = -self.k / pos**2
        # TODO: Expression pour l'accélération angulaire en coordonnées cylindriques
        B[1] = 0
        return B

    def Em_cyl(self, A) :
        # TODO: Expression pour l'énergie mécanique en coordonnées cylindriques
        r = A[1]
        v2 = A[2]**2 * r**2 + A[3]**2
        Em = 0.5 * v2 - self.k / r
        return Em

    def L_cyl(self, A) :
        # TODO: Expression pour la quantité de mouvement angulaire en coordonnées cylindriques
        r = A[1]
        L = r**2 * A[2]
        return L

    # Cartésiennes

    def der_A_cart(self, t, A):
        B = np.empty(4)

        # Coordonnées polaires
        x, y, vx, vy = A[0], A[1], A[2], A[3]

        # Calcul de la distance radiale r
        r = np.sqrt(x**2 + y**2)

        # Expressions des dérivées par rapport au temps en coordonnées cartésiennes
        B[0] = vx
        B[1] = vy
        B[2] = self.acc_cart(t, [x, y], [vx, vy])[0]
        B[3] = self.acc_cart(t, [x, y], [vx, vy])[1]

        return B

    def acc_cart(self, t, pos, vel):
        B = np.empty(2)

        # Calcul du carré de la distance
        r_squared = pos[0]**2 + pos[1]**2

        # Calcul de l'accélération en coordonnées cartésiennes
        B[0] = -self.k * pos[0] / r_squared**(3/2)
        B[1] = -self.k * pos[1] / r_squared**(3/2)

        return B

    def Em_cart(self, A) :
        # TODO: Expression pour l'énergie mécanique en coordonnées cartésiennes
        r = np.sqrt(A[0]**2 + A[1]**2)
        v2 = A[2]**2 + A[3]**2
        Em = 0.5 * v2 - self.k / r
        return Em

    def L_cart(self, A) :
        # TODO: Expression pour la quantité de mouvement angulaire en coordonnées cartésiennes
        r = np.sqrt(A[0]**2 + A[1]**2)
        L = r**2 * np.sqrt(A[2]**2 + A[3]**2)
        return L



    """Fonctions pour une résolution bien plus mathématique"""
    # Attention ici on doit juste résoudre une équation différentielle d'ordre 1
    # On utilise la forme elliptique de la solution. Il est plus facile de calculer
    # le mouvement avec précision si l'on connait la trajectoire !
    
    def r(self, theta):
        C,e,p,theta_p = self.C, self.e, self.p, self.theta_p
        return p/(1+e*np.cos(theta - theta_p))
    
    def der_theta(self, t, theta) :
        C,e,p,theta_p = self.C, self.e, self.p, self.theta_p
        return C / self.r(theta)**2
    
    def der_r(self, theta):
        C,e,p,theta_p = self.C, self.e, self.p, self.theta_p
        return -p*e*self.der_theta(0.0, theta)*np.sin(theta - theta_p)/(1+e*np.cos(theta - theta_p))**2

    def A_ref(self, theta_ref):
        N = theta_ref.size
        A = np.empty((N,4))
        for i in range(N):
            A[i] = self.A_theta(theta_ref[i])
        return A

    def A_theta(self, theta) :
        e,p,k = self.e, self.p, self.k
        A = np.empty(4)
        A[0] = theta
        A[1] = self.r(theta)
        A[2] = self.der_theta(0.0, theta)
        A[3] = self.der_r(theta)
        return A
    
    def Em_theta(self, theta) :
        e,p,k = self.e, self.p, self.k
        r = self.r(theta)
        dertheta = self.der_theta(0.0, theta)
        derr = self.der_r(theta)
        v2 = derr**2+(r*dertheta)**2
        Em = 1/2*v2-k/r
        return Em

    def L_theta(self, theta) :
        e,p = self.e, self.p
        r = self.r(theta)
        dertheta = self.der_theta(0.0, theta)
        L = r**2*dertheta
        return L