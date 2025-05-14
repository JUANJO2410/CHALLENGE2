import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import plotly.express as px

# 1. Configuración de página y estilo con fondo negro
st.set_page_config(page_title="Dashboard Bajas – TelecomX", layout="wide")
BAR_COLORS = ['#FF6138', '#FFFF9D', '#BEEB9F']  # naranja, amarillo, verde claro

st.markdown("<style>body {background-color: #000000; color: white;}</style>", unsafe_allow_html=True)
st.title("📉 Dashboard de Bajas – TelecomX")

# 2. Carga y preparación de datos
LOCAL_JSON = Path(r"C:\Users\juanj\OneDrive\Python\Challenge2\TelecomX_Data.json")
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if LOCAL_JSON.exists():
        return pd.read_json(LOCAL_JSON)
    url = ("https://raw.githubusercontent.com/alura-cursos/"
           "challenge2-data-science-LATAM/main/TelecomX_Data.json")
    return pd.read_json(url)

# Función para normalizar estructuras anidadas
def flatten_df(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    df = df.copy()
    # Explode listas
    for col in [c for c in df.columns if df[c].apply(lambda x: isinstance(x, list)).any()]:
        df = df.explode(col, ignore_index=True)
    # Normalizar diccionarios
    while True:
        dict_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x, dict)).any()]
        if not dict_cols:
            break
        for col in dict_cols:
            norm = pd.json_normalize(df[col]).add_prefix(f"{col}{sep}")
            df = pd.concat([df.drop(columns=[col]).reset_index(drop=True), norm.reset_index(drop=True)], axis=1)
    return df

@st.cache_data(show_spinner=True)
def prep_dataset() -> pd.DataFrame:
    df = flatten_df(load_data())
    df["account_Charges.Total"] = df["account_Charges.Total"].replace(r"^\s*$", np.nan, regex=True).astype(float)
    rename_cols = {
        'customerID':'ID_CLIENTE','Churn':'BAJA','customer_gender':'GENERO_CLIENTE',
        'customer_SeniorCitizen':'CLIENTE_SENIOR','customer_tenure':'ANTIGUEDAD_CLIENTE',
        'internet_InternetService':'SERVICIO_INET','account_Contract':'CONTRATO',
        'account_PaymentMethod':'METODO_PAGO','account_Charges.Monthly':'CARGO_MENSUAL',
        'account_Charges.Total':'CARGO_TOTAL','internet_OnlineSecurity':'SEGURIDAD_ONLINE',
        'internet_OnlineBackup':'RESPALDO_ONLINE','internet_DeviceProtection':'PROTECCION_DISPOSITIVOS',
        'internet_TechSupport':'SOPORTE_TECNICO','internet_StreamingTV':'STREAMING',
        'internet_StreamingMovies':'PELICULAS_STREAMING'
    }
    df.rename(columns=rename_cols, inplace=True, errors="ignore")
    df['BAJA'] = df['BAJA'].map({'Yes':1,'No':0})
    return df

# Prepara data
df = prep_dataset()
df_clean = df.dropna(subset=['BAJA'])

# 3. Sidebar: filtros
st.sidebar.header("🎛️ Filtros")
mask = pd.Series(True, index=df_clean.index)
for label, col in {'Género':'GENERO_CLIENTE','Contrato':'CONTRATO','Internet':'SERVICIO_INET'}.items():
    opts = sorted(df_clean[col].dropna().unique())
    sel = st.sidebar.multiselect(label, opts, default=[])
    if sel:
        mask &= df_clean[col].isin(sel)

df_filt = df_clean[mask]

# 4. KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Clientes", f"{len(df_filt):,}", delta_color="normal")
col2.metric("Bajas", f"{df_filt['BAJA'].sum():,}", f"{df_filt['BAJA'].mean()*100:.1f}%", delta_color="inverse")
col3.metric("ARPU (US$)", f"{df_filt['CARGO_MENSUAL'].mean():.2f}")
st.divider()

# 5. Gráficos con pestañas usando Plotly
tab_intro, tab1, tab2, tab3 = st.tabs(
    ["ℹ️ Introducción", "Género / Senior", "Fidelidad", "Cruces"]
)
# -------- Pestaña 0: explicación general --------------------------
with tab_intro:
    st.subheader("¿Qué estás viendo en este dashboard?")
    st.markdown(
        """
        Este tablero resume el **comportamiento de bajas (churn)** en TelecomX:

        * **KPIs principales** arriba (clientes, bajas, ARPU).
        * **Género / Senior**: distribución de bajas por género y por clientes senior.
        * **Fidelidad**: relación entre antigüedad y probabilidad de baja.
        * **Cruces**: análisis dinámico de variables contractuales vs. bajas.
        * **Modelo predictivo** debajo: LightGBM + ROC-AUC y descarga de clientes de alto riesgo.
        
        Ajusta los filtros de la barra lateral para ver cómo cambian los indicadores.
        """
    )
    st.info(
        "Tip: utiliza el control «Variable cruzada» para detectar segmentos "
        "con mayor % de bajas y priorizar acciones de retención."
    )
with tab1:
    colA, colB = st.columns(2)
    # Género
    df_gender = df_filt.groupby(['GENERO_CLIENTE','BAJA']).size().reset_index(name='Clientes')
    fig = px.bar(
        df_gender,
        x='GENERO_CLIENTE', y='Clientes', color='BAJA', barmode='stack',
        color_discrete_sequence=BAR_COLORS,
        labels={'GENERO_CLIENTE':'Género','BAJA':'Baja'}
    )
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='black', plot_bgcolor='black',
        font_color='white', legend_title_text='Baja'
    )
    colA.plotly_chart(fig, use_container_width=True)

    # Senior
    df_senior = df_filt.groupby(['CLIENTE_SENIOR','BAJA']).size().reset_index(name='Clientes')
    fig2 = px.bar(
        df_senior,
        x='CLIENTE_SENIOR', y='Clientes', color='BAJA', barmode='stack',
        color_discrete_sequence=BAR_COLORS,
        labels={'CLIENTE_SENIOR':'Senioridad','BAJA':'Baja'}
    )
    fig2.update_layout(
        template='plotly_dark', paper_bgcolor='black', plot_bgcolor='black',
        font_color='white', legend_title_text='Baja'
    )
    colB.plotly_chart(fig2, use_container_width=True)

with tab2:
    bins = [0,12,24,48, df_filt['ANTIGUEDAD_CLIENTE'].max()+1]
    df_aux = df_filt.copy()
    df_aux['FIDELIDAD'] = pd.cut(
        df_aux['ANTIGUEDAD_CLIENTE'], bins=bins,
        labels=['≤12 m','13-24 m','25-48 m','≥49 m'], right=False
    )
    df_fid = df_aux.groupby(['FIDELIDAD','BAJA']).size().reset_index(name='Clientes')
    fig3 = px.bar(
        df_fid,
        x='FIDELIDAD', y='Clientes', color='BAJA', barmode='stack',
        color_discrete_sequence=BAR_COLORS,
        labels={'FIDELIDAD':'Fidelidad','BAJA':'Baja'}
    )
    fig3.update_layout(
        template='plotly_dark', paper_bgcolor='black', plot_bgcolor='black',
        font_color='white', legend_title_text='Baja'
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    var = st.selectbox('Variable cruzada',['SERVICIO_INET','CONTRATO','METODO_PAGO'])
    cross = (
        df_filt.groupby([var,'BAJA'])['ID_CLIENTE']
        .count().unstack(fill_value=0)
        .rename(columns={0:'Activos',1:'Bajas'})
    )
    cross['%Bajas'] = cross['Bajas']/cross.sum(axis=1)*100
    st.dataframe(cross.style.format({'%Bajas':'{:.1f}%'}))

# 6. Comparativa ARPU
crit = (
    (df_clean['ANTIGUEDAD_CLIENTE']<=12)&
    (df_clean['CONTRATO']=='Month-to-month')&
    (df_clean['SERVICIO_INET']=='Fiber optic')&
    (df_clean['METODO_PAGO']=='Electronic check')
)
plot_df = pd.concat([
    df_clean[df_clean['BAJA']==0].assign(grupo='Activos'),
    df_clean[(df_clean['BAJA']==1)&(~crit)].assign(grupo='Otras Bajas'),
    df_clean[crit].assign(grupo='Segmento crítico')
])
fig4 = px.box(
    plot_df, x='grupo', y='CARGO_MENSUAL', color='grupo',
    color_discrete_sequence=BAR_COLORS,
    labels={'grupo':'Grupo','CARGO_MENSUAL':'US$'}
)
fig4.update_layout(
    template='plotly_dark', paper_bgcolor='black', plot_bgcolor='black',
    font_color='white', showlegend=False
)
st.plotly_chart(fig4, use_container_width=True)

# 7. Modelo Predictivo
st.divider()
st.header('⚙️ Modelo Predictivo')
with st.expander('Entrenar / actualizar modelo'):
    thr = st.slider('Umbral de alerta (prob ≥ …)',0.1,0.9,0.4,0.05)
    if st.button('Entrenar modelo'):
        feats = ['CONTRATO','SERVICIO_INET','METODO_PAGO','ANTIGUEDAD_CLIENTE','CLIENTE_SENIOR',
                 'SEGURIDAD_ONLINE','RESPALDO_ONLINE','PROTECCION_DISPOSITIVOS','STREAMING','PELICULAS_STREAMING',
                 'CARGO_MENSUAL']
        X, y = df_clean[feats].copy(), df_clean['BAJA']
        cats = X.select_dtypes('object').columns; X[cats]=X[cats].astype('category')
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)
        model = LGBMClassifier(n_estimators=400,learning_rate=0.05,subsample=0.9,
                               colsample_bytree=0.9,objective='binary')
        model.fit(X_tr,y_tr,categorical_feature=list(cats))
        y_prob = model.predict_proba(X_te)[:,1]
        st.success(f"ROC-AUC: {roc_auc_score(y_te,y_prob):.3f}")
        st.write('Matriz de confusión', confusion_matrix(y_te,(y_prob>thr).astype(int)))
        df_clean['p_baja'] = model.predict_proba(X)[:,1]
        df_clean['ALTO_RIESGO'] = (df_clean['p_baja']>thr).astype(int)
        csv = df_clean[df_clean['ALTO_RIESGO']==1][['ID_CLIENTE','p_baja']]
        csv = csv.to_csv(index=False).encode()
        st.download_button('📥 Descargar alto riesgo',csv,'alto_riesgo.csv','text/csv')

st.caption('© 2025 — Juan José Ramos • Streamlit + LightGBM')
