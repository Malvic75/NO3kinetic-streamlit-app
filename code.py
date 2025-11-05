# app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 📌 Fonctions cinétiques
def f1(x, Ka, n, Vm):
    return Vm * (x**n) / (Ka**n + x**n)

def f2(x, Ka, n, a, Vm):
    return np.where(x <= 0.5, a * x, Vm * (x**n) / (Ka**n + x**n))

def f2_smooth(x, Ka, n, a, Vm, x_switch, delta):
    sigmoid = 1 / (1 + np.exp(-(x - x_switch) / delta))
    return (1 - sigmoid) * a * x + sigmoid * Vm * (x**n) / (Ka**n + x**n)

def f3(x, Ka, n, a, Vm):
    return a * x + Vm * (x**n) / (Ka**n + x**n)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

# 📊 Données simulées
x1 = np.linspace(0.01, 2, 50)
y1 = f1(x1, 1.5, 2.0, 1.0) + np.random.normal(0, 0.05, len(x1))
y2 = f2(x1, 1.2, 1.8, 0.3, 1.0) + np.random.normal(0, 0.05, len(x1))

# ⚙️ Ajustement vitesse_app
Vm1 = max(y1)
popt1, _ = curve_fit(lambda x, Ka, n: f1(x, Ka, n, Vm1), x1, y1, p0=[1, 1], bounds=(0, 25))
Ka1, n1 = popt1

# 📈 Interface Streamlit
st.title("🔬 Ajustement de paramètres cinétiques")

st.subheader("1️⃣ Ajustement de `vitesse_app`")
Ka_slider = st.slider("Ka", min_value=0.1, max_value=10.0, value=float(Ka1), step=0.1)
n_slider = st.slider("n", min_value=0.1, max_value=5.0, value=float(n1), step=0.05)

x_vals = np.linspace(min(x1), max(x1), 300)
y_fit = f1(x_vals, Ka_slider, n_slider, Vm1)
r2_score = r2(y1, f1(x1, Ka_slider, n_slider, Vm1))

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].scatter(x1, y1, label='Données', color='green')
axs[0].plot(x_vals, y_fit, label=f'Fit\nKa={Ka_slider:.3f}, n={n_slider:.3f}\nR²={r2_score:.4f}', color='blue')
axs[0].set_title('Échelle linéaire')
axs[0].set_xlabel('x'); axs[0].set_ylabel('v_app')
axs[0].grid(True); axs[0].legend()

axs[1].scatter(x1, y1, label='Données', color='green')
axs[1].plot(x_vals, y_fit, label=f'Fit\nKa={Ka_slider:.3f}, n={n_slider:.3f}\nR²={r2_score:.4f}', color='blue')
axs[1].set_xscale('log'); axs[1].set_yscale('log')
axs[1].set_title('Échelle log-log')
axs[1].set_xlabel('x'); axs[1].set_ylabel('v_app')
axs[1].grid(True, which='both'); axs[1].legend()

st.pyplot(fig)

# 🔁 Tu peux répéter ce schéma pour les autres équations (`f2`, `f2_smooth`, `f3`) en ajoutant des sliders pour leurs paramètres respectifs et en affichant les courbes ajustées.

# 📋 Résumé des paramètres
params = pd.DataFrame({
    'Équation': ['vitesse_app'],
    'Vm': [Vm1],
    'Ka': [Ka1],
    'n': [n1],
    'a': [np.nan],
    'R²': [r2_score],
    'RMSE': [rmse(y1, f1(x1, Ka1, n1, Vm1))],
})
st.subheader("📋 Résumé des paramètres")
st.dataframe(params)
