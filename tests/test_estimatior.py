"""
Suite de tests riche pour `LabelNoiseEstimator`.

But: ce fichier est conçu comme une suite "riche" et exploratoire de tests
il génère des graphiques et sauvegarde des figures. Les commentaires ci-dessous expliquent l'objectif
de chaque test et les étapes réalisées.

Usage: lancer ce script en standalone pour reproduire les figures de
diagnostic. Les fonctions commencent par `test_...` mais sont destinées
à une exécution manuelle / interactive plutôt qu'à `pytest` automatique.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from src.data_generation import generate_data, eta_function
from src.estimatior import LabelNoiseEstimator

# --- Configuration globale pour les plots et la reproductibilité ---
plt.style.use('ggplot')  # style graphique cohérent pour toutes les figures
np.random.seed(42)      # graine fixe pour résultats reproductibles


def calculate_mse(eta_true, eta_hat):
    """Retourne la Mean Squared Error entre `eta_true` et `eta_hat`.

    Paramètres:
    - eta_true: array-like, valeurs « vraies » de η(x) sur une grille
    - eta_hat:  array-like, valeurs prédites par l'estimateur

    Renvoie:
    - float: MSE moyenne
    """
    return np.mean((eta_true - eta_hat) ** 2)


def test_visual_fit():
    """Test visuel (1D) pour comparer η vraie et η estimée.

    Pour chaque type de bruit (linéaire, sinus), on :
    1) génère un jeu de données (X, Y, Z, η_true),
    2) entraîne `LabelNoiseEstimator` sur (X, Z),
    3) prédit η sur une grille fine, calcule la MSE,
    4) trace η vraie vs η estimée et sauvegarde la figure.

    Notes:
    - Les points de flip (`Z != Y`) sont affichés faiblement pour
      indiquer où les erreurs d'étiquetage se produisent.
    - Le but est diagnostique : montrer la qualité d'ajustement visuel.
    """
    print("TEST 1: Visual Fit Performance (1D)")

    noise_types = ['linear', 'sine']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i, noise in enumerate(noise_types):
        # --- Génération des données ---
        # X : variables d'entrée, Y : labels propres, Z : labels bruités
        X, Y, Z, eta_true = generate_data(n=1500, d=1, noise_type=noise)

        # --- Estimation ---
        # Choix d'une fenêtre h typique pour 1D
        model = LabelNoiseEstimator(h=0.2)
        model.fit(X, Z)

        # --- Évaluation sur une grille pour un tracé lisse ---
        x_grid = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
        eta_hat = model.predict_eta(x_grid)
        eta_real = eta_function(x_grid, noise_type=noise)

        # --- Métrique simple pour suivi numérique ---
        mse = calculate_mse(eta_real, eta_hat)
        print(f"Model: {noise:6s} | MSE: {mse:.6f}")

        # --- Tracés ---
        axes[i].plot(x_grid, eta_real, 'g-', lw=3, label='True η(x)')
        axes[i].plot(x_grid, eta_hat, 'r--', lw=2, label='Estimated η_hat(x)')
        # Points de flips affichés en petite opacité
        axes[i].scatter(X, (Z != Y).astype(float) * 0.4, alpha=0.03, color='blue', label='Actual Flips')
        axes[i].set_title(f"Noise Type: {noise.upper()} (MSE: {mse:.5f})")
        axes[i].legend()
        axes[i].set_ylim(-0.05, 0.5)

    plt.tight_layout()
    plt.savefig('tests/results_visual_fit.png')
    print("-> Plot saved: tests/results_visual_fit.png\n")


def test_bandwidth_impact():
    """Analyse du rôle de la fenêtre `h` (biais-variance).

    On compare trois valeurs de `h` pour illustrer:
    - trop petit => sous-lissage, variance élevée,
    - intermédiaire => équilibre,
    - trop grand => sur-lissage, biais élevé.
    """
    print("TEST 2: Bandwidth (h) Impact Analysis")

    X, Y, Z, eta_true = generate_data(n=1000, d=1, noise_type='sine')
    h_values = [0.02, 0.2, 2.0]
    titles = ['Under-smoothing (High Var)', 'Optimal h', 'Over-smoothing (High Bias)']

    x_grid = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    eta_real = eta_function(x_grid, noise_type='sine')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, h in enumerate(h_values):
        # Instanciation de l'estimateur avec la fenêtre courante
        model = LabelNoiseEstimator(h=h)
        model.fit(X, Z)
        eta_hat = model.predict_eta(x_grid)

        mse = calculate_mse(eta_real, eta_hat)
        print(f"h: {h:4.2f} | {titles[i]:25s} | MSE: {mse:.6f}")

        axes[i].plot(x_grid, eta_real, 'g-', lw=2)
        axes[i].plot(x_grid, eta_hat, 'r--', lw=2)
        axes[i].set_title(f"h = {h} ({titles[i]})")
        axes[i].set_ylim(-0.05, 0.5)

    plt.tight_layout()
    plt.savefig('tests/results_bandwidth.png')
    print("-> Plot saved: tests/results_bandwidth.png\n")


def test_convergence_m():
    """Étude de la décroissance de l'erreur en fonction de la taille d'échantillon m.

    On trace la MSE en échelle log-log pour différentes tailles d'échantillon
    afin d'observer la tendance de convergence empirique.
    """
    print("TEST 3: Convergence Analysis (Influence of m)")

    m_sizes = [100, 200, 500, 1000, 2000, 5000]
    errors = []

    # Grille d'évaluation fixe pour comparer les modèles
    X_eval = np.linspace(-2, 2, 200).reshape(-1, 1)
    eta_real = eta_function(X_eval, noise_type='sine')

    for m in m_sizes:
        X, Y, Z, _ = generate_data(n=m, d=1, noise_type='sine')
        model = LabelNoiseEstimator(h=0.2)
        model.fit(X, Z)
        eta_hat = model.predict_eta(X_eval)

        err = calculate_mse(eta_real, eta_hat)
        errors.append(err)
        print(f"Sample size m: {m:5d} | MSE: {err:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(m_sizes, errors, 'bo-', lw=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Convergence of Estimation Error")
    plt.xlabel("Sample Size (m)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, which="both", ls="-")
    plt.savefig('tests/results_convergence.png')
    print("-> Plot saved: tests/results_convergence.png\n")


def test_curse_of_dimensionality():
    """Étude empirique de l'effet de la dimension (curse of dimensionality).

    Pour des dimensions croissantes, on observe généralement une dégradation
    des performances d'estimation pour des méthodes nonparamétriques.
    """
    print("TEST 4: Curse of Dimensionality (Influence of d)")

    dimensions = [1, 2, 3, 5, 8, 10, 20, 50]
    errors = []

    for d in dimensions:
        # Pour garder un coût raisonnable, on fixe n=2000
        X, Y, Z, eta_true = generate_data(n=2000, d=d, noise_type='linear')
        # On augmente h pour atténuer partiellement l'effet de grande dimension
        model = LabelNoiseEstimator(h=0.5)
        model.fit(X, Z)

        # Évaluation sur les mêmes points : simple et informatif pour la tendance
        eta_hat = model.predict_eta(X)
        err = calculate_mse(eta_true, eta_hat)
        errors.append(err)
        print(f"Dimension d: {d:2d} | MSE: {err:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, errors, 'ro-', lw=2)
    plt.title("Impact of Dimension d on Estimation Error")
    plt.xlabel("Dimension (d)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.savefig('tests/results_dimension.png')
    print("-> Plot saved: tests/results_dimension.png\n")


if __name__ == "__main__":
    # Exécution séquentielle des tests riches (diagnostiques)
    start_time = time.time()
    print("\n" + "#" * 60)
    print("RUNNING RICH ESTIMATION SUITE")
    print("#" * 60 + "\n")

    test_visual_fit()
    test_bandwidth_impact()
    test_convergence_m()
    test_curse_of_dimensionality()

    duration = time.time() - start_time
    print("#" * 60)
    print(f"ALL TESTS COMPLETED IN {duration:.2f}s")
    print("#" * 60)