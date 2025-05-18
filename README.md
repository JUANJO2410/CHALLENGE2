# TelecomX LATAM – Análisis Exploratorio de Bajas

> Análisis de fuga (churn) en clientes de telecomunicaciones de Latinoamérica utilizando **Python**, **Pandas** y **LightGBM**. Este repositorio incluye un _notebook_ de Jupyter con todo el flujo de trabajo de EDA, así como un **dashboard interactivo en Streamlit** para la visualización ejecutiva de resultados.

---

## 📂 Contenido del repositorio

| Ruta/archivo | Descripción |
|--------------|-------------|
| `TelecomX_LATAM.ipynb` | Notebook principal con preparación de datos, análisis exploratorio, segmentación y entrenamiento del modelo predictivo. |
| `Informe.md` | Informe con los hallazgos significativos y soluciones para el cliente. |
| `app_churn.py` | Aplicación Streamlit que consume el dataset procesado y el modelo LightGBM para explorar métricas y descargar clientes en riesgo. |
| `TelecomX_Data.json` | Dataset original en formato JSON (estructuras anidadas). |
| `requirements.txt` | Dependencias mínimas de Python. |

---

## 1. Descripción del problema

Las compañías de telecomunicaciones afrontan pérdidas significativas debido a la **fuga de clientes (churn)**. El objetivo de este proyecto es entender los **factores que impulsan las bajas** y construir un modelo que anticipe clientes con alta probabilidad de abandonar el servicio.

---

## 2. Flujo de trabajo

1. **Ingesta & Normalización**  
   * Se lee el JSON con **Pandas**, se *explodean* listas y se **normalizan diccionarios anidados** con `pd.json_normalize`, generando un _flat table_ listo para análisis.
2. **Limpieza & Enriquecimiento**  
   * Conversión de valores vacíos a `NaN` y tipado correcto (`float`, `category`, etc.).  
   * Renombrado de columnas a español y mapeo de la variable `Churn` a binaria (1 = Baja, 0 = Activo).
3. **Análisis Exploratorio (EDA)**  
   * Estadísticos descriptivos y distribución de **KPIs** (charges, tenure, servicios).  
   * Visualización de churn por **género**, **senioridad**, **contrato**, **método de pago** y **servicio de internet**.  
   * **Segmentación crítica**: clientes con ≤12 meses de antigüedad, contrato _month‑to‑month_, fibra óptica y pago por _electronic check_.
4. **Hallazgos clave**  
   * Los clientes _month‑to‑month_ tienen una tasa de bajas > **43 %**.  
   * El método de pago _electronic check_ duplica el churn frente a tarjetas/crédito automático.  
   * El **segmento crítico** concentra **ARPU alto** y 3× probabilidad de baja, por lo que retenerlo impacta directamente en ingresos.
5. **Modelado Predictivo**  
   * Se entrena un **LightGBMClassifier** con variables contractuales, de servicio y de comportamiento de uso.  
   * Métrica principal: **ROC‑AUC ≈ 0.81** y matriz de confusión para un umbral configurable (default 0.40).
6. **Dashboard Ejecutivo**  
   * KPIs en tiempo real, gráficos interactivos de Plotly y pestañas para Género/Senior, Fidelidad, Cruces, una Introducción y Conclusiones. Publicado en Streamlit Community Cloud.
   * Descarga de CSV con clientes **de alto riesgo** identificados por el modelo.

---

## 3. Ejecución local

```bash
# 1. Clonar el repo
$ git clone https://github.com/JUANJO2410/CHALLENGE2.git
$ cd CHALLENGE2

# 2. Crear entorno (opcional)
$ python -m venv .venv && source .venv/bin/activate

# 3. Instalar dependencias
$ pip install -r requirements.txt

# 4. Abrir el notebook
$ jupyter lab TelecomX_LATAM.ipynb

# 5. Lanzar el dashboard
$ streamlit run app_churn.py --server.port 8501



```


## 4. Estructura del notebook

| Sección                     | Celdas | Descripción                                                                           |
|-----------------------------|:------:|---------------------------------------------------------------------------------------|
| **1. Setup & Librerías**    | 1-6    | Imports, configuración de estilo y funciones auxiliares.                              |
| **2. Carga de datos**       | 7-12   | Descarga/lectura del JSON y **flatten** de estructuras.                               |
| **3. Limpieza**             | 13-20  | Tratamiento de `NaN`, casteo y renombrado.                                            |
| **4. EDA**                  | 21-40  | Visualizaciones univariadas y bivariadas en Plotly.                                   |
| **5. Segmentación crítica** | 41-46  | Identificación y cuantificación del segmento de mayor riesgo.                         |
| **6. Modelado**             | 47-60  | Split train/test, entrenamiento LightGBM y evaluación ROC-AUC & matriz confusión.      |
| **7. Exportes**             | 61-65  | Guardado de predicciones y dataset enriquecido.                                       |

---

## 5. Créditos

Proyecto desarrollado por **Juan José Ramos** (BIM & Data Science).  
juan.ramos.s@usach.cl

---


