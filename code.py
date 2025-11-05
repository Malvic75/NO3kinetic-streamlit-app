import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Fonctions cinétiques
def f1(x, Ka, n, Vm):
    return Vm * (x**n) / (Ka**n + x**n)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

# Données simulées
x1 = np.linspace(0.01, 2, 50)
y1 = f1(x1, 1.5, 2.0, 1.0) + np.random.normal(0, 0.05, len(x1))

# Ajustement
Vm1 = max(y1)
popt1, _ = curve_fit(lambda x, Ka, n: f1(x, Ka, n, Vm1), x1, y1, p0=[1, 1], bounds=(0, 25))
Ka1, n1 = popt1

# Interface Streamlit
st.title("🔬 Ajustement de paramètres cinétiques")
st.subheader("1️⃣ Équation : vitesse_app")

Ka = st.slider("Ka", 0.1, 10.0, float(Ka1), 0.1)
n = st.slider("n", 0.1, 5.0, float(n1), 0.05)

x_vals = np.linspace(min(x1), max(x1), 300)
y_fit = f1(x_vals, Ka, n, Vm1)
r2_score = r2(y1, f1(x1, Ka, n, Vm1))

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].scatter(x1, y1, label='Données', color='green')
axs[0].plot(x_vals, y_fit, label=f'Fit\nKa={Ka:.3f}, n={n:.3f}\nR²={r2_score:.4f}', color='blue')
axs[0].set_title('Échelle linéaire')
axs[0].set_xlabel('x'); axs[0].set_ylabel('v_app')
axs[0].grid(True); axs[0].legend()

axs[1].scatter(x1, y1, label='Données', color='green')
axs[1].plot(x_vals, y_fit, label=f'Fit\nKa={Ka:.3f}, n={n:.3f}\nR²={r2_score:.4f}', color='blue')
axs[1].set_xscale('log'); axs[1].set_yscale('log')
axs[1].set_title('Échelle log-log')
axs[1].set_xlabel('x'); axs[1].set_ylabel('v_app')
axs[1].grid(True, which='both'); axs[1].legend()

st.pyplot(fig)

# Résumé
st.markdown("### 📋 Résumé des paramètres")
st.write(f"→ Vm = {Vm1:.4f}")
st.write(f"→ Ka = {Ka:.4f}")
st.write(f"→ n  = {n:.4f}")
st.write(f"→ R² = {r2_score:.4f}")
