# Dashboard GDPR - Proyecto TFM (Optimizado para MacBook Pro M2)

![Optimizado para M2 Pro](https://img.shields.io/badge/optimizado-M2%20Pro-purple.svg)
![macOS](https://img.shields.io/badge/macOS-15.4.1-orange.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

Este dashboard interactivo está específicamente optimizado para tu **MacBook Pro 14" (2023) con chip M2 Pro, 16GB de RAM y macOS 15.4.1**. Permite analizar y visualizar técnicas de anonimización y privacidad sobre un dataset de transacciones financieras, facilitando la comprensión de conceptos como K-anonimato, L-diversidad y Privacidad Diferencial (DP), sin requerir conocimientos avanzados.

## 🚀 Características Optimizadas para MBP 2023

- **Carga optimizada de datos**: Utiliza PyArrow para carga más rápida y menor consumo de memoria
- **Reducción automática de precisión**: Convierte automáticamente tipos de datos para maximizar eficiencia
- **Paralelización controlada**: Limita el número de hilos para mejor rendimiento y duración de batería
- **Visualizaciones Retina**: Gráficos de alta resolución optimizados para tu pantalla Liquid Retina XDR
- **Límites de memoria**: Controla el consumo de RAM para evitar ralentizaciones en análisis de grandes volúmenes

## 📊 Contenido del Dataset

El archivo `dataset/informacion_anonimizacion.csv` contiene las siguientes columnas:

| Columna           | Descripción                                                 |
|-------------------|------------------------------------------------------------|
| step              | Identificador de paso temporal                              |
| type              | Tipo de transacción (PAYMENT, TRANSFER, CASH_OUT, etc.)     |
| amount            | Monto de la transacción                                     |
| nameOrig          | Identificador del cliente que origina la transacción        |
| oldbalanceOrg     | Saldo anterior del cliente que origina la transacción       |
| newbalanceOrig    | Saldo posterior del cliente que origina la transacción      |
| nameDest          | Identificador del destinatario                              |
| oldbalanceDest    | Saldo anterior del destinatario                             |
| newbalanceDest    | Saldo posterior del destinatario                            |
| isFraud           | 1 si la transacción es fraudulenta, 0 si no                 |
| isFlaggedFraud    | 1 si la transacción fue marcada como potencial fraude       |

## 🔧 Requisitos del Sistema

Estos requisitos están específicamente optimizados para tu hardware:

- **Hardware** (ya disponible en tu equipo):
  - MacBook Pro 14" (2023) con Apple M2 Pro
  - 16GB de memoria unificada
  - macOS 15.4.1

- **Software** (a instalar):
  - Python 3.11+ (optimizado para ARM)
  - Bibliotecas Python optimizadas para Apple Silicon

## ⚙️ Instalación Optimizada

1. **Crear entorno virtual optimizado para MBP 2023**:
   ```bash
   # Abre Terminal
   python3 -m venv venv_m2 --system-site-packages
   source venv_m2/bin/activate
   ```

2. **Instalar dependencias optimizadas**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements_m2.txt
   ```

3. **Verificar optimización para MBP 2023**:
   ```bash
   python -c "import numpy as np; print(f'NumPy detecta {np.__config__.show_config()}')"
   ```

## 🚀 Ejecución del Dashboard

1. **Activar entorno virtual**:
   ```bash
   source venv_m2/bin/activate
   ```

2. **Ejecutar dashboard con optimizaciones para MBP 2023**:
   ```bash
   streamlit run dashboard_gdpr.py
   ```

3. **Para mejor rendimiento, ejecutar con optimizaciones específicas**:
   ```bash
   OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 streamlit run dashboard_gdpr.py
   ```

## 📱 Características del Dashboard

### 1. Resumen Ejecutivo
- **Métricas clave**: K-anonimato, L-diversidad, Épsilon (DP)
- **Indicadores de riesgo**: Evaluación automática del nivel de protección
- **Comparativa de rendimiento**: Visualización del impacto en precisión y sensibilidad
- **Logs de auditoría**: Registro de acciones y cambios para trazabilidad GDPR

### 2. Técnicas de Anonimización
- **K-anonimato**: Visualización interactiva de tamaños de grupo y cumplimiento
- **L-diversidad**: Análisis de diversidad por grupo y verificación de cumplimiento
- **Privacidad Diferencial**: Simulación del impacto del parámetro Épsilon

### 3. Desempeño del Modelo
- **Matrices de confusión**: Comparativa antes/después de anonimización
- **Importancia de variables**: Análisis del cambio en relevancia de predictores
- **Curva ROC**: Evaluación del poder predictivo con diferentes configuraciones
- **Evaluación de impacto**: Análisis cuantitativo y cualitativo de la relación privacidad-utilidad

## 🔍 Explicación de Conceptos Clave

### K-Anonimato
El K-anonimato garantiza que cada registro sea indistinguible de al menos K-1 otros registros, mediante la agrupación de datos similares. Por ejemplo, con K=10, cada cliente aparece en un grupo con al menos 9 otros clientes con características similares, dificultando la identificación individual.

### L-Diversidad
La L-diversidad asegura que dentro de cada grupo K-anónimo exista suficiente variedad en los atributos sensibles. Por ejemplo, con L=2, cada grupo debe tener al menos 2 tipos diferentes de transacciones, evitando inferencias sobre comportamientos específicos.

### Privacidad Diferencial (DP)
Añade "ruido" matemático controlado a los datos o al proceso de entrenamiento, asegurando que los resultados no cambien significativamente por la inclusión o exclusión de un único registro. El parámetro Épsilon (ε) controla el nivel de privacidad: menor ε = mayor privacidad pero menor utilidad.

## 📋 Perfiles de Rendimiento para MBP 2023

El dashboard incluye tres perfiles optimizados para tu hardware:

1. **Alto Rendimiento** (con alimentación):
   - Utiliza más núcleos (8+)
   - Gráficos de alta resolución (200-300 DPI)
   - Ideal para análisis detallados

2. **Balanceado** (batería + rendimiento):
   - Utiliza núcleos intermedios (4-6)
   - Gráficos de calidad media (150-200 DPI)
   - Equilibrio entre rendimiento y duración de batería

3. **Eficiencia Energética** (máxima duración de batería):
   - Utiliza menos núcleos (2-4)
   - Gráficos optimizados para bajo consumo (100-150 DPI)
   - Prioriza duración de batería sobre rendimiento

## ⚠️ Notas importantes

- Para un rendimiento óptimo en tu MBP 2023, se recomienda limitar datasets a menos de 5 millones de filas o usar el formato parquet en lugar de CSV.
- Si experimentas problemas de memoria, activa el perfil de Eficiencia Energética desde la interfaz.
- La ejecución completa del dashboard requiere aproximadamente 200-400MB de RAM, muy por debajo de los 16GB disponibles en tu hardware.
- Para datasets significativamente grandes, considera convertirlos a formato parquet:
  ```python
  import pandas as pd
  df = pd.read_csv('dataset/informacion_anonimizacion.csv')
  df.to_parquet('dataset/informacion_anonimizacion.parquet', engine='pyarrow', compression='zstd')
  ```

## 📞 Soporte y Contacto

Si encuentras algún problema específico con tu MacBook Pro M2 o necesitas ayuda, no dudes en contactarme.