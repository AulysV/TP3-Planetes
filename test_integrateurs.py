# -*- coding: utf-8 -*-
"""
@author: Y. Vadée Le Brun

DESCRIPTION

Ce code propose plusieurs résolutions numériques du pendule plan.
Une quantification de l'erreur numérique et une étude énergétique sont abordés.
"""

"""
BIBLIOTHEQUES
"""
# import de la bibliothèque numpy (gestion de matrices et routines mathématiques) en lui donnant le surnom np
import numpy as np
# import de la bibliothèque matplotlib (graphiques) en lui donnant le surnom plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import de la bibliothèque time qui permet de mesurer le temps d'éxécution d'un programme
import time
# import du module integrate de la bibliothèque scipy qui dispose d'un integrateur de référence : odeint
from scipy.integrate import odeint

"""
BIBLIOTHEQUES PERSONNELLES
"""
# import des intégrateurs numériques présents dans integrateur_complet
from integrateur_complet import *
from integrateur_meca import *
from planete import Planet

"""
FONCTIONS PERSONNELLES
"""
# Comme la solution exacte n'est pas disponible, on calcule une solution de référence
# avec odeint en lui demandant une grande précision
# On utilise de plus, des astuces mathématiques afin d'augmenter la précision.
def calcul_planete_ref(Terre_cyl):
    theta_ref = odeint(Terre_cyl.der_theta, Terre_cyl.CI()[0], temps, tfirst = True)
    A_ref = Terre_cyl.A_ref(theta_ref)
    em_ref = Terre_cyl.Em_theta(theta_ref)
    L_ref = Terre_cyl.L_theta(theta_ref)
    return A_ref, em_ref, L_ref

# Cette fonction résoud numériquement le mouvement de la planète avec différents intégrateurs possibles
# Pour chaque intégrateur, il faut préciser : sa couleur d'affichage et le pas de temps souhaité
# Elle ajoute au dictionnaire :
# La solution
# L'erreur par rapport à une solution de référence
# L'énergie mécanique en fonction du temps
def calculs_planete(solver_list, planete, A_ref, temps):
    for item in solver_list:
        solver_class, dt = item["solver_class"], item["dt"]
        # On mesure le temps d'exécution avec la bibliothèque time
        start = time.perf_counter()
        # Pour le calcul de la solution, la syntaxe d'odeint est différente de la notre
        # il faut donc prévoir une disjonction de cas pour si on l'utilise
        if solver_class == "Odeint":
            A = odeint(planete.derA, planete.CI(), temps, tfirst = True)
            solver_class_name = "odeint"
        else:
            solver = solver_class(planete)
            A = solver.solve(planete.CI(), temps, dt)
            print(A)
            solver_class_name = solver_class.__name__
        end = time.perf_counter()
        elapsed = (end - start) * 1000
        print("Le calcul via", solver_class_name, "a duré", elapsed,"ms")

        # Calcul de l'erreur à chaque pas de temps
        if planete.cyl:
            A_x = A[:,1]*np.cos(A[:,0])
            A_y = A[:,1]*np.sin(A[:,0])
        else:
            A_x = A[:,0]
            A_y = A[:,1]

        A_ref_x = A_ref[:,1]*np.cos(A_ref[:,0])
        A_ref_y = A_ref[:,1]*np.sin(A_ref[:,0])
        erreur = np.sqrt((A_x - A_ref_x)**2 + (A_y - A_ref_y)**2)
    
        # L'erreur maximale est le maximum de ce tableau
        erreur_max = max(erreur)
        print("L'erreur via", solver_class_name, "vaut", erreur_max)
        # Calcul de l'énergie mécanique
        em = planete.Em(A.T)
        # On ajoute les trois grandeurs calculées au dictionnaire pour pouvoir y accéder plus tard
        item["A"] = A
        item["Ax"] = A_x
        item["Ay"] = A_y
        item["erreur"] = erreur
        item["em"] = em

def plot_trajectoire(solver_list, A_ref):
    xt = A_ref[:,1]*np.cos(A_ref[:,0])
    yt = A_ref[:,1]*np.sin(A_ref[:,0])
    plt.plot(xt,yt,"-k", lw=1.0, label="résolution mathématique")
    for item in solver_list:
        if item["solver_class"] == "Odeint": solver_name = "Odeint"
        else: solver_name = item["solver_class"].__name__
        plt.plot(item["Ax"], item["Ay"], item["color"], lw=1.0, label=solver_name)
    plt.legend(loc='lower right')
    plt.axis('scaled')
    plt.title("Trajectoire de la Terre autour du soleil")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)

def plot_erreur(solver_list):
    for item in solver_list:
        if item["solver_class"] == "Odeint": solver_name = "Odeint"
        else: solver_name = item["solver_class"].__name__
        plt.plot(temps, item["erreur"], item["color"], lw=1.0, label=solver_name)
    plt.legend(loc='lower right')
    plt.title("Erreur en distance au cours du temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Erreur (m)")
    plt.grid(True)

def plot_em(solver_list, em_ref):
    plt.plot(temps, em_ref, "-k", lw=1.0, label="résolution mathématique")
    for item in solver_list:
        if item["solver_class"] == "Odeint": solver_name = "Odeint"
        else: solver_name = item["solver_class"].__name__
        plt.plot(temps, item["em"], item["color"], lw=1.0, label=solver_name)
    plt.legend(loc='lower right')
    plt.title("Energie mécanique de la Terre au cours du temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Energie mécanique (J/kg)")
    plt.grid(True)

def update_frame(frame, ax, solver_list, xt, yt, scatters, titre):
    time_in_years = temps[frame]/(3600*24*365)
    title = f'Temps: {time_in_years:.3f} ans'
    titre.set_text(title)
    for i, item in enumerate(solver_list):
        scatters[i].set_offsets([item["Ax"][frame], item["Ay"][frame]])

    scatters[-1].set_offsets([xt[frame], yt[frame]])
    return scatters + [titre]

def animate_trajectories(fig, ax, solver_list, A_ref):
    # Dessin de la trajectoire de référence
    xt = A_ref[:, 1] * np.cos(A_ref[:, 0])
    yt = A_ref[:, 1] * np.sin(A_ref[:, 0])
    ax.plot(xt, yt, "-k", lw=1.0, label="Reference")

    # Dessins de la planète sous forme de point pour chaque intégrateur + la référence
    scatters = []
    for i, item in enumerate(solver_list):
        solver_class = item["solver_class"]
        if solver_class == "Odeint": solver_name = "odeint"
        else:                        solver_name = solver_class.__name__
        scatters.append(ax.scatter(item["Ax"][0], item["Ay"][0], c = item["color"], label = solver_name))
    scatters.append(ax.scatter(xt[0], yt[0], c = 'black', label = "reference"))
    # Et le soleil pour faire joli ;)
    ax.scatter(0, 0, c = 'gold', s = 100)

    ax.legend(loc='lower right')
    ax.set_title("Trajectoire")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)
    plt.axis('scaled')

    title = "Temps: " + f'{0}'
    titre = ax.text(0.5, 0.5, '', transform=ax.transAxes, ha="center")

    ani = animation.FuncAnimation(fig, update_frame, frames=len(temps), fargs=(ax, solver_list, xt, yt, scatters, titre), interval=50, blit = True)
    plt.show()

"""
CODE PRINCIPAL
"""

""" INITIALISATION ET DEFINITION DES PARAMETRES DE SIMULATION"""
# durée de la simulation (réglée sur 4 ans en secondes)
t_max = 3600*24*365*10
# Nombre de points voulu pour l'affichage
N = 10010

# vitesse angulaire initiale de la Terre
# Valeur donnée en supposant le mouvement circulaire : 2.0*np.pi/(365*3600*24)
w_0 = 2.0*np.pi/(365*3600*24) / 2

# Booléens pour les options du programme
# Etude en cartésiennes
bool_cart = True
# Etude en cylindriques
bool_cyl = False
# Figure animée
bool_ani = True

# Dictionnaire des intégrateurs étudiés, de la couleur de tracé et du pas de temps utilisé pour chacun
#{"solver_class": ForwardEuler, "color": "-r", "dt": 1e-7} pas utilisé car trop lent ...
"""
solver_list = [{"solver_class": ExplicitMidpoint, "color": "-b", "dt": 6e-5},
               {"solver_class": RungeKutta4, "color": "-g", "dt": 9e-3},
               {"solver_class": VelocityVerlet, "color": "-c", "dt": 6e-5},
               {"solver_class": Stormer_Verlet, "color": "-y", "dt": 6e-5},
               {"solver_class": "Odeint", "color": "-m", "dt": 1e-3}  ]
               """
dt = 3600 * 24
solver_list_cart = [{"solver_class": RungeKutta4, "color": 'green', "dt": dt * 4},
               {"solver_class": ExplicitMidpoint, "color": 'blue', "dt": dt},
               {"solver_class": MecaVelocityVerlet, "color": 'red', "dt": dt},
               {"solver_class": "Odeint", "color": 'magenta', "dt": 1e-3}]

solver_list_cyl = [{"solver_class": RungeKutta4, "color": 'green', "dt": dt * 4},
               {"solver_class": ExplicitMidpoint, "color": 'blue', "dt": dt},
               {"solver_class": "Odeint", "color": 'magenta', "dt": 1e-3}]

"""INITIALISATION VARIABLES DE CALCULS"""
# Liste des temps auquels sauvegarder les résultats pour les afficher
temps = np.linspace(0, t_max, N)
# Création du pendule
Terre_cart = Planet(omega_0 = w_0, cyl = False)
Terre_cyl = Planet(omega_0 = w_0, cyl = True)

"""Calcul de la solution de référence"""
A_ref, em_ref, L_ref = calcul_planete_ref(Terre_cyl)

if bool_cart:
    """ETUDE EN CARTESIENNES"""
    """Calculs"""
    # Calcul du mouvement via les intégrateurs développés par nous
    calculs_planete(solver_list_cart, Terre_cart, A_ref, temps)

    """Figure pour la trajectoire"""
    plt.figure("Trajectoire cartesiennes")
    plot_trajectoire(solver_list_cart, A_ref)

    """Figure pour l'erreur en fonction du temps"""
    plt.figure("Erreur cartesiennes")
    plot_erreur(solver_list_cart)

    """Figure pour l'énergie mécanique en fonction du temps"""
    plt.figure("Energie cartesiennes")
    plot_em(solver_list_cart, em_ref)

if bool_cyl:
    """ETUDE EN CYLINDRIQUES"""
    """Calculs"""
    # Calcul du mouvement via les intégrateurs développés par nous
    calculs_planete(solver_list_cyl, Terre_cyl, A_ref, temps)

    """Figure pour la trajectoire"""
    plt.figure("Trajectoire cylindriques")
    plot_trajectoire(solver_list_cyl, A_ref)

    """Figure pour l'erreur en fonction du temps"""
    plt.figure("Erreur cylindriques")
    plot_erreur(solver_list_cyl)

    """Figure pour l'énergie mécanique en fonction du temps"""
    plt.figure("Energie cylindriques")
    plot_em(solver_list_cyl, em_ref)

if not bool_ani: plt.show()

if bool_ani:
    """ Animation """
    fig = plt.figure("Trajectoire")
    ax = plt.subplot()
    if bool_cart and not bool_cyl:
        solver_list = solver_list_cart
    elif not bool_cart and bool_cyl:
        solver_list = solver_list_cyl
    else:
        solver_list = solver_list_cart + solver_list_cyl

    animate_trajectories(fig, ax, solver_list, A_ref)

    