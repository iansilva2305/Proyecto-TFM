# Dashboard GDPR - Proyecto TFM

Este dashboard interactivo permite analizar y visualizar técnicas de anonimización y privacidad sobre un dataset de transacciones financieras, facilitando la comprensión de conceptos como K-anonimato, L-diversidad y Privacidad Diferencial (DP), sin requerir conocimientos avanzados.

## Contenido del Dataset

El archivo `dataset/informacion_anonimizacion.csv` contiene las siguientes columnas:

| Columna           | Descripción                                                                                  |
|-------------------|---------------------------------------------------------------------------------------------|
| step              | Identificador de paso temporal (por ejemplo, hora o ciclo de simulación)                    |
| type              | Tipo de transacción (PAYMENT, TRANSFER, CASH_OUT, DEBIT, etc.)                              |
| amount            | Monto de la transacción                                                                     |
| nameOrig          | Identificador del cliente que origina la transacción                                        |
| oldbalanceOrg     | Saldo anterior del cliente que origina la transacción                                       |
| newbalanceOrig    | Saldo posterior del cliente que origina la transacción                                      |
| nameDest          | Identificador del destinatario                                                              |
| oldbalanceDest    | Saldo anterior del destinatario                                                             |
| newbalanceDest    | Saldo posterior del destinatario                                                            |
| isFraud           | 1 si la transacción es fraudulenta, 0 si no                                                 |
| isFlaggedFraud    | 1 si la transacción fue marcada como potencial fraude por el sistema, 0 si no               |

**Ejemplo de una fila del dataset:**

| step | type     | amount   | nameOrig    | oldbalanceOrg | newbalanceOrig | nameDest    | oldbalanceDest | newbalanceDest | isFraud | isFlaggedFraud |
|------|----------|----------|-------------|---------------|----------------|-------------|----------------|----------------|---------|----------------|
| 1    | PAYMENT  | 9839.64  | C1231006815 | 170136.0      | 160296.36      | M1979787155 | 0.0            | 0.0            | 0       | 0              |

---

**Visualización del dataset en la app:**

La pestaña "Vista General" muestra las primeras filas de la tabla para que puedas explorar los datos fácilmente.

---

## ¿Qué hace el Dashboard?

La aplicación permite:

1. **Explorar el dataset**: Visualiza una muestra de las transacciones y métricas básicas.
2. **Configurar parámetros de anonimización y privacidad**: Ajusta desde la barra lateral los valores de:
    - **K-Anonimato**: Tamaño mínimo de grupo para que un individuo no sea identificable.
    - **L-Diversidad**: Diversidad mínima de valores sensibles en cada grupo.
    - **Épsilon (DP)**: Nivel de privacidad diferencial aplicado (simulado).
3. **Visualizar técnicas de anonimización**:
    - Histograma de tamaños de grupo para K-anonimato.
    - Tabla de verificación de L-diversidad, resaltando los grupos que no cumplen.
4. **Analizar el rendimiento del modelo**:
    - Métricas de precisión con y sin anonimización.
    - Matriz de confusión y curva ROC (simuladas).
5. **Auditoría**: Sección de logs ficticios para simular acciones relevantes.

## Explicación de los conceptos y pasos del algoritmo

### 1. K-Anonimato
- Agrupa las transacciones por el campo `nameOrig` (cliente emisor).
- Calcula el tamaño de cada grupo (cuántas transacciones realizó cada cliente).
- Un grupo cumple K-anonimato si su tamaño es mayor o igual al valor K seleccionado.

**Ejemplo visual:**

Supón que tienes estos datos:

| nameOrig     | ... |  
|--------------|-----|  
| C111         | ... |  
| C111         | ... |  
| C222         | ... |  
| C333         | ... |  
| C333         | ... |  
| C333         | ... |  

Si K=2: 
- C111 y C333 cumplen K-anonimato (aparecen al menos 2 veces).
- C222 no cumple (solo aparece una vez).

**En la app:**
- El histograma muestra cuántos clientes tienen 1, 2, 3... transacciones.
- Una línea roja indica el valor de K elegido.

![Ejemplo K-anonimato](assets/ejemplo_k_anonimato.png)
> Imagen generada automáticamente para fines ilustrativos.

---

### 2. L-Diversidad
- Para cada grupo (por `nameOrig`), cuenta cuántos valores diferentes hay en la columna `type` (tipo de transacción).
- Un grupo cumple L-diversidad si tiene al menos L valores distintos.

**Ejemplo visual:**

| nameOrig | type      |
|----------|-----------|
| C111     | PAYMENT   |
| C111     | TRANSFER  |
| C111     | PAYMENT   |
| C222     | PAYMENT   |
| C222     | PAYMENT   |

Si L=2:
- C111 cumple L-diversidad (tiene PAYMENT y TRANSFER).
- C222 no cumple (solo PAYMENT).

**En la app:**
- La tabla muestra para cada grupo cuántos tipos diferentes tiene y si cumple L-diversidad.
- Los grupos que no cumplen se resaltan en rojo.

![Ejemplo L-diversidad](assets/ejemplo_l_diversidad.png)
> Imagen generada automáticamente para fines ilustrativos.

---

### 3. Privacidad Diferencial (DP)
- El parámetro Épsilon permite simular el nivel de privacidad diferencial aplicado durante el entrenamiento del modelo.
- Un valor bajo de Épsilon implica mayor privacidad, pero menor utilidad de los datos (y viceversa).
- En este dashboard, la aplicación de DP es conceptual/simulada.

**Ejemplo visual:**

```
Privacidad alta (ε bajo):
  Datos muy protegidos, pero el modelo puede perder precisión.
Privacidad baja (ε alto):
  Datos menos protegidos, pero el modelo es más preciso.
```

Puedes ajustar ε en la barra lateral y ver cómo cambia la explicación en la pestaña correspondiente.

---

### 4. Métricas del modelo
- Simula la predicción de fraude y calcula métricas como precisión, matriz de confusión y curva ROC.
- Permite comparar el rendimiento del modelo con y sin anonimización.

**Ejemplo visual:**

- **Matriz de confusión:**

|        | Predicción: No Fraude | Predicción: Fraude |
|--------|----------------------|--------------------|
| Real: No Fraude |    1000              |      10           |
| Real: Fraude    |     5                |      50           |

- **Curva ROC:**

Una gráfica donde se compara la tasa de verdaderos positivos vs. falsos positivos. El área bajo la curva (AUC) indica la calidad del modelo.

![Ejemplo ROC](assets/ejemplo_roc.png)
> Imagen generada automáticamente para fines ilustrativos.

---

### 5. Auditoría
- Muestra logs de acciones relevantes para simular trazabilidad y cumplimiento normativo.

**Ejemplo visual:**

| Acción                 | Usuario | Fecha       | Detalle                        |
|------------------------|---------|-------------|--------------------------------|
| Entrenamiento modelo   | admin   | 2024-05-01  | Entrenado con DP ε=2.0         |
| Cambio parámetro K     | admin   | 2024-05-02  | K cambiado a 10                |

Estos logs aparecen en la pestaña "Auditoría y Logs".

---

## ¿Cómo usar la app?

1. Instala las dependencias necesarias:

   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta el dashboard con:
   ```bash
   streamlit run dashboard_gdpr.py
   ```
3. Ajusta los parámetros desde la barra lateral y explora las diferentes pestañas para ver cómo afectan los resultados.

## Notas importantes
- El archivo de datos (`informacion_anonimizacion.csv`) es grande y **no se incluye en el repositorio**. Puedes solicitarlo al autor o cargar tu propio archivo en la carpeta `dataset/`.
- Si el archivo no está presente, la app mostrará un mensaje de error y se detendrá hasta que lo agregues.
- El dashboard está diseñado para ser educativo y conceptual, no para producción real.

---

Si tienes dudas o sugerencias, ¡no dudes en abrir un issue o contactar al autor!
