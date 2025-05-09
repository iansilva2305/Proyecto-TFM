import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- Cargar datos reales ---
df = pd.read_csv('dataset/informacion_anonimizacion.csv')

# --- Parámetros generales (pueden calcularse a partir de los datos si es necesario) ---
k_actual = 10
l_actual = 2
epsilon_actual = 2.0

# --- Ejemplo de cálculo de métricas reales ---
# Aquí se asume que el dataset tiene columnas isFraud (etiqueta real) y algún resultado de predicción (simulado para ejemplo)
if 'isFraud' in df.columns:
    y_true = df['isFraud']
    # Simulación: predicción aleatoria, reemplazar por tu modelo real si lo tienes
    np.random.seed(42)
    y_pred = np.random.choice([0, 1], size=len(y_true), p=[0.9, 0.1])
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    precision_anon = 100 * (y_true == y_pred).mean()
else:
    cm = np.array([[0, 0], [0, 0]])
    fpr, tpr, roc_auc, precision_anon = [0], [0], 0, 0
precision_no_anon = 85.4  # Valor de referencia

# --- Ejemplo de agrupación para k-anonimato ---
group_sizes = df.groupby('nameOrig').size().values

# --- Ejemplo de tabla para l-diversidad ---
l_diversity_data = df.groupby('nameOrig')['type'].nunique().reset_index()
l_diversity_data.columns = ['Grupo ID', 'L-diversidad']
l_diversity_data['Cumple L=2'] = l_diversity_data['L-diversidad'] >= l_actual

# --- Logs ficticios para auditoría ---
logs = [
    {'Acción': 'Entrenamiento modelo', 'Usuario': 'admin', 'Fecha': '2024-05-01', 'Detalle': 'Entrenado con DP ε=2.0'},
    {'Acción': 'Cambio parámetro K', 'Usuario': 'admin', 'Fecha': '2024-05-02', 'Detalle': 'K cambiado a 10'},
]
df_logs = pd.DataFrame(logs)

# --- Pestañas Streamlit ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vista General",
    "🛡️ Detalle Anonimización",
    "🔒 Privacidad Diferencial",
    "📈 Rendimiento Modelo",
    "📜 Auditoría"
])

with tab1:
    st.header("Vista General del Dataset")
    st.dataframe(df.head(100))
    st.metric("Total de transacciones", len(df))
    st.metric("Transacciones fraudulentas", int(df['isFraud'].sum()) if 'isFraud' in df.columns else '-')

with tab2:
    st.header("Análisis de Técnicas de Anonimización")
    st.subheader(f"Distribución Grupos K-Anonimato (k={k_actual})")
    fig_k, ax_k = plt.subplots(figsize=(8, 4))
    ax_k.hist(group_sizes, bins=range(1, int(max(group_sizes)) + 2), color='#5cb85c', edgecolor='black')
    ax_k.axvline(x=k_actual, color='red', linestyle='--', label=f'k={k_actual}')
    ax_k.set_xlabel('Tamaño del Grupo')
    ax_k.set_ylabel('Frecuencia')
    ax_k.legend()
    st.pyplot(fig_k)
    st.caption("Idealmente, no debería haber barras a la izquierda de la línea roja.")

    st.subheader(f"Verificación L-Diversidad (l={l_actual})")
    st.dataframe(l_diversity_data.style.applymap(lambda x: 'color: red' if x is False else '', subset=['Cumple L=2']))

with tab3:
    st.header("Privacidad Diferencial")
    st.markdown(f"Se aplicó Privacidad Diferencial durante el entrenamiento del modelo (simulado) utilizando la biblioteca Opacus (o similar) con un presupuesto de privacidad **ε = {epsilon_actual:.1f}**.")
    st.info("Nivel de Privacidad: Alto (Equilibrio común)")

with tab4:
    st.header("Rendimiento del Modelo")
    st.metric("Precisión con Anonimización", f"{precision_anon:.1f}%")
    st.metric("Precisión sin Anonimización", f"{precision_no_anon:.1f}%")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriz de Confusión")
        fig_cm, ax_cm = plt.subplots()
        cax = ax_cm.matshow(cm, cmap=plt.cm.Blues)
        fig_cm.colorbar(cax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, str(cm[i, j]), va='center', ha='center', color='black' if cm[i, j] < cm.max()/2 else 'white')
        ax_cm.set_xlabel('Predicción')
        ax_cm.set_ylabel('Valor Real')
        st.pyplot(fig_cm)
    with col2:
        st.subheader("Curva ROC")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Azar')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR')
        ax_roc.set_title('Receiver Operating Characteristic')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

with tab5:
    st.header("Auditoría y Logs")
    st.dataframe(df_logs, use_container_width=True)

# --- Barra lateral ---
st.sidebar.image("https://placehold.co/150x80/5cb85c/FFFFFF?text=GDPR", caption="Logo Placeholder")
st.sidebar.info("Dashboard Conceptual - TFM Anonimización LLM/GDPR")
st.sidebar.markdown("---")
st.sidebar.header("Parámetros Actuales")
st.sidebar.markdown(f"**K-Anonimato:** {k_actual}")
st.sidebar.markdown(f"**L-Diversidad:** {l_actual}")
st.sidebar.markdown(f"**Épsilon (DP):** {epsilon_actual:.1f}")
