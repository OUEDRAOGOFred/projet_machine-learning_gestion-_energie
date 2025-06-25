import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
import random
import os
import datetime

# --- STYLING ---
st.markdown("""
    <style>
    /* --- General Body --- */
    body {
        background-color: var(--background-color);
    }
    
    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
    }
    .sidebar-logo {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1em;
    }
    .sidebar-logo img {
        width: 60px;
        border-radius: 50%;
        box-shadow: 0 2px 12px #00c6ff44;
        border: 2px solid #00c6ff;
    }
    .sidebar-title {
        font-size: 1.7em;
        font-weight: bold;
        background: linear-gradient(90deg, #00c6ff, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5em;
        text-align: center;
    }
    
    /* --- Accueil Page Specific --- */
    .gradient-title {
        font-size: 3.2em;
        font-weight: bold;
        background: linear-gradient(90deg, #00c6ff, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2em;
        text-align: center;
    }
    .slogan-box {
        background: var(--secondary-background-color);
        border-radius: 12px;
        padding: 1em 2em;
        font-size: 1.3em;
        color: #0072ff;
        margin-bottom: 1.5em;
        box-shadow: 0 2px 12px #0000001a;
        text-align: center;
    }
    .feature-card {
        background: var(--secondary-background-color);
        border-radius: 16px;
        box-shadow: 0 4px 24px #0000001a;
        padding: 1.5em;
        margin: 0.5em;
        text-align: center;
        transition: transform 0.2s;
        color: var(--text-color) !important;
    }
    .feature-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 8px 32px #00c6ff44;
    }
    .feature-icon {
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .feature-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #0072ff !important;
        margin-bottom: 0.2em;
    }
    .feature-desc {
        font-size: 1em;
        color: #333 !important;
    }
    
    /* --- Introduction Box --- */
    .intro-box {
        background-color: var(--secondary-background-color);
        border-left: 5px solid #0072ff;
        border-radius: 8px;
        padding: 1em 1.5em;
        margin: 2em 0;
        box-shadow: 0 2px 12px #0000001a;
    }
    .intro-box h3 {
        color: #0072ff;
        margin-bottom: 0.5em;
        font-size: 1.5em;
    }
    .intro-box p {
        line-height: 1.6;
    }
    
    /* --- Conclusion Box --- */
    .conclusion-box {
        background-color: var(--secondary-background-color);
        border-left: 5px solid #28a745; /* Green for success */
        border-radius: 8px;
        padding: 1.5em 2em;
        margin-top: 2em;
        box-shadow: 0 2px 12px #0000001a;
    }
    .conclusion-box h3 {
        color: #28a745;
        margin-bottom: 0.7em;
        font-size: 1.5em;
        border-bottom: 2px solid #28a74533;
        padding-bottom: 0.3em;
    }
    .conclusion-box ul {
        list-style-type: '✅  ';
        padding-left: 20px;
    }
    .conclusion-box li {
        margin-bottom: 0.5em;
        line-height: 1.6;
    }
    
    /* --- Page Headers --- */
    .page-header {
        font-size: 2.2em;
        font-weight: bold;
        color: var(--text-color);
        border-bottom: 3px solid #0072ff;
        padding-bottom: 0.3em;
        margin-bottom: 1em;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Gestion intelligente de la consommation énergétique",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo"><img src="https://cdn-icons-png.flaticon.com/512/3103/3103476.png" alt="logo"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    menu = st.radio(
        "Aller à :",
        [
            "🏠 Accueil",
            "🔎 Exploration des données",
            "📈 Modélisation",
            "🔮 Prédiction Personnalisée",
            "🧩 Clustering et profils",
            "🚨 Détection d'anomalies",
            "🤖 Optimisation (Q-learning)",
            "✅ Conclusion"
        ]
    )

# --- DATA LOADING ---
@st.cache_data
def load_data():
    csv_path = "donnees.csv"
    xlsx_path = "donnees.xlsx"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    else:
        st.error("❌ Fichier de données introuvable ! veuillez entrer le bon chemin d'accès")
        st.stop()
        
    df['date'] = pd.to_datetime(df['date'])
    df.drop(columns=["rv1", "rv2"], inplace=True, errors='ignore')
    
    # Feature engineering
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    def get_period(hour):
        if 5 <= hour < 12: return 'matin'
        elif 12 <= hour < 17: return 'après-midi'
        elif 17 <= hour < 22: return 'soir'
        else: return 'nuit'
    df['period'] = df['hour'].apply(get_period)
    df['period_code'] = df['period'].map({'nuit': 0, 'matin': 1, 'après-midi': 2, 'soir': 3})
    
    temp_cols = [col for col in df.columns if col.startswith("T") and not col.startswith("T_out")]
    df["temp_mean"] = df[temp_cols].mean(axis=1)
    hum_cols = [col for col in df.columns if col.startswith("RH_") and col not in ['RH_out']]
    df["hum_mean"] = df[hum_cols].mean(axis=1)
    df["confort_index"] = df["temp_mean"] - (0.55 * (1 - df["hum_mean"] / 100) * (df["temp_mean"] - df["Tdewpoint"]))
    df["Appliances_roll_mean_3"] = df["Appliances"].rolling(window=3, min_periods=1).mean()
    df["T_out_roll_mean_6"] = df["T_out"].rolling(window=6, min_periods=1).mean()
    
    return df

df = load_data()

@st.cache_resource
def get_model(dataf):
    """Entraîne et retourne le modèle XGBoost."""
    features = [
        'hour', 'day_of_week', 'weekend', 'period_code',
        'temp_mean', 'hum_mean', 'confort_index',
        'T_out', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint',
        'Appliances_roll_mean_3', 'T_out_roll_mean_6'
    ]
    X = dataf[features]
    y = dataf['Appliances']
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X, y)
    return model

model = get_model(df)

# --- PAGE ROUTING ---

# 1. Accueil
if menu == "🏠 Accueil":
    st.markdown('<div class="gradient-title">Gestion intelligente de la consommation énergétique</div>', unsafe_allow_html=True)
    st.markdown('<div class="slogan-box">Un tableau de bord interactif pour explorer, modéliser et optimiser la consommation énergétique grâce à l\'intelligence artificielle.</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="intro-box">
            <h3>Contexte du Projet</h3>
            <p>La gestion intelligente de la consommation énergétique est un enjeu majeur dans un contexte de transition énergétique, de hausse des coûts et de recherche d'efficacité domestique. À travers ce projet, nous exploitons un jeu de données provenant de capteurs installés dans une habitation résidentielle en Belgique, combiné à des données météorologiques horaires issues d'une station locale.</p>
            <p>Notre objectif est de concevoir un système capable non seulement de prédire la consommation énergétique à partir des conditions environnementales, mais également de comprendre les habitudes de consommation et de détecter des comportements anormaux ou inefficaces.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.image(
        "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=900&q=80",
        caption="L'intelligence artificielle appliquée à l'énergie (crédit : Unsplash)",
        use_container_width=True
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">⚡</div>'
            '<div class="feature-title">Analyse intelligente</div>'
            '<div class="feature-desc">Visualisez et comprenez vos données énergétiques en un clin d\'œil.</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🤖</div>'
            '<div class="feature-title">Prédiction & Optimisation</div>'
            '<div class="feature-desc">Des modèles IA avancés pour anticiper et réduire la consommation.</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🔔</div>'
            '<div class="feature-title">Détection d\'anomalies</div>'
            '<div class="feature-desc">Repérez automatiquement les comportements inhabituels ou inefficaces.</div>'
            '</div>',
            unsafe_allow_html=True
        )

# 2. Exploration des données
elif menu == "🔎 Exploration des données":
    st.markdown('<h1 class="page-header">🔎 Exploration des données</h1>', unsafe_allow_html=True)
    st.subheader("Aperçu du dataset")
    st.dataframe(df.head(20))
    st.markdown(f"**Nombre de lignes :** {df.shape[0]} | **Nombre de colonnes :** {df.shape[1]}")
    st.subheader("Consommation moyenne par heure")
    hourly = df.groupby('hour')['Appliances'].mean()
    fig1 = px.line(x=hourly.index, y=hourly.values, labels={'x':'Heure','y':'Consommation moyenne (Wh)'})
    st.plotly_chart(fig1, use_container_width=True)
    st.subheader("Consommation moyenne par jour de la semaine")
    days = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    weekly = df.groupby('day_of_week')['Appliances'].mean()
    fig2 = px.bar(x=days, y=weekly.values, labels={'x':'Jour','y':'Consommation moyenne (Wh)'})
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Consommation moyenne par mois")
    monthly = df.groupby('month')['Appliances'].mean()
    fig3 = px.bar(x=monthly.index, y=monthly.values, labels={'x':'Mois','y':'Consommation moyenne (Wh)'})
    st.plotly_chart(fig3, use_container_width=True)
    st.subheader("Évolution de la consommation dans le temps")
    fig4 = px.line(x=df['date'], y=df['Appliances'], labels={'x':'Date','y':'Consommation (Wh)'})
    st.plotly_chart(fig4, use_container_width=True)
    st.subheader("Matrice de corrélation")
    corr = df.corr(numeric_only=True)
    fig5 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu', origin='lower')
    st.plotly_chart(fig5, use_container_width=True)

# 3. Modélisation
elif menu == "📈 Modélisation":
    st.markdown('<h1 class="page-header">📈 Modélisation de la consommation</h1>', unsafe_allow_html=True)
    features = [
        'hour', 'day_of_week', 'weekend', 'period_code',
        'temp_mean', 'hum_mean', 'confort_index',
        'T_out', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint',
        'Appliances_roll_mean_3', 'T_out_roll_mean_6'
    ]
    X = df[features]
    y = df['Appliances']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.markdown("Choisissez un modèle :")
    model_choice = st.selectbox("Modèle", ["Régression linéaire", "Random Forest", "XGBoost"])
    
    if model_choice == "Régression linéaire":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.success(f"MAE : {mae:.2f} | RMSE : {rmse:.2f} | R² : {r2:.3f}")
    
    st.subheader("Comparaison des valeurs réelles et prédites (échantillon)")
    fig6 = px.line(x=np.arange(200), y=[y_test.values[:200], y_pred[:200]], labels={'x':'Index','value':'Consommation (Wh)'},
                  color_discrete_sequence=["#636EFA", "#EF553B"])
    fig6.data[0].name = 'Vrai'
    fig6.data[1].name = 'Prédit'
    st.plotly_chart(fig6, use_container_width=True)
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("Importance des variables")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances}).sort_values(by='importance', ascending=False)
        fig7 = px.bar(imp_df, x='importance', y='feature', orientation='h')
        st.plotly_chart(fig7, use_container_width=True)

elif menu == "🔮 Prédiction Personnalisée":
    st.markdown('<h1 class="page-header">🔮 Prédiction de Consommation Personnalisée</h1>', unsafe_allow_html=True)

    # --- 1. Initialisation de l'état ---
    # On s'assure que toutes les clés nécessaires existent au début.
    if 'pred_initiated' not in st.session_state:
        st.session_state.pred_date = datetime.date.today()
        st.session_state.pred_time = datetime.time(12, 0)
        st.session_state.pred_temp_mean = 21.0
        st.session_state.pred_hum_mean = 40.0
        st.session_state.pred_t_out = 10.0
        st.session_state.pred_rh_out = 70.0
        st.session_state.pred_windspeed = 10.0
        st.session_state.pred_visibility = 20.0
        st.session_state.pred_tdewpoint = 5.0
        st.session_state.pred_app_roll_mean = 60.0
        st.session_state.pred_t_out_roll_mean = 10.0
        st.session_state.prediction_result = None
        st.session_state.avg_for_hour_result = None
        st.session_state.hour_of_prediction_result = None
        st.session_state.pred_initiated = True

    # --- 2. Fonctions de callback ---
    def make_prediction():
        # Cette fonction est appelée quand on clique sur "Lancer la Prédiction"
        # Elle lit les valeurs actuelles des widgets (via st.session_state) et fait le calcul.
        try:
            hour = st.session_state.pred_time.hour
            day_of_week = st.session_state.pred_date.weekday()
            weekend = 1 if day_of_week >= 5 else 0
            
            period_map = {h: 'matin' for h in range(5, 12)}
            period_map.update({h: 'après-midi' for h in range(12, 17)})
            period_map.update({h: 'soir' for h in range(17, 22)})
            period = period_map.get(hour, 'nuit')
            period_code = {'nuit': 0, 'matin': 1, 'après-midi': 2, 'soir': 3}[period]
            
            confort_index = st.session_state.pred_temp_mean - (0.55 * (1 - st.session_state.pred_hum_mean / 100) * (st.session_state.pred_temp_mean - st.session_state.pred_tdewpoint))

            input_data = pd.DataFrame([{
                'hour': hour, 'day_of_week': day_of_week, 'weekend': weekend, 'period_code': period_code,
                'temp_mean': st.session_state.pred_temp_mean, 'hum_mean': st.session_state.pred_hum_mean, 'confort_index': confort_index,
                'T_out': st.session_state.pred_t_out, 'RH_out': st.session_state.pred_rh_out, 'Windspeed': st.session_state.pred_windspeed,
                'Visibility': st.session_state.pred_visibility, 'Tdewpoint': st.session_state.pred_tdewpoint,
                'Appliances_roll_mean_3': st.session_state.pred_app_roll_mean, 'T_out_roll_mean_6': st.session_state.pred_t_out_roll_mean
            }])

            prediction = model.predict(input_data)[0]
            st.session_state.prediction_result = prediction
            st.session_state.avg_for_hour_result = df[df['hour'] == hour]['Appliances'].mean()
            st.session_state.hour_of_prediction_result = hour
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")

    def reset_form():
        # Cette fonction efface les résultats et réinitialise le formulaire pour une nouvelle prédiction
        st.session_state.pred_date = datetime.date.today()
        st.session_state.pred_time = datetime.time(12, 0)
        st.session_state.pred_temp_mean = 21.0
        st.session_state.pred_hum_mean = 40.0
        st.session_state.pred_t_out = 10.0
        st.session_state.pred_rh_out = 70.0
        st.session_state.pred_windspeed = 10.0
        st.session_state.pred_visibility = 20.0
        st.session_state.pred_tdewpoint = 5.0
        st.session_state.pred_app_roll_mean = 60.0
        st.session_state.pred_t_out_roll_mean = 10.0
        st.session_state.prediction_result = None
        st.session_state.avg_for_hour_result = None
        st.session_state.hour_of_prediction_result = None

    # --- 3. Affichage du formulaire ---
    # Le bouton pour commencer une nouvelle prédiction est en dehors du formulaire.
    st.button("🔄 Nouvelle Prédiction", on_click=reset_form, help="Réinitialise le formulaire pour de nouvelles valeurs.")

    with st.form(key='prediction_form'):
        st.subheader("🗓️ Date et Heure")
        st.date_input("Date", key='pred_date')
        st.time_input("Heure", key='pred_time')
        
        st.subheader("🏠 Conditions Intérieures")
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Température intérieure moyenne (°C)", -10.0, 40.0, key='pred_temp_mean', step=0.1)
        with col2:
            st.slider("Humidité intérieure moyenne (%)", 0.0, 100.0, key='pred_hum_mean', step=1.0)

        st.subheader("🌦️ Conditions Extérieures")
        col3, col4, col5 = st.columns(3)
        with col3:
            st.slider("Température extérieure (°C)", -20.0, 50.0, key='pred_t_out', step=0.1)
            st.slider("Humidité extérieure (%)", 0.0, 100.0, key='pred_rh_out', step=1.0)
        with col4:
            st.slider("Vitesse du vent (km/h)", 0.0, 80.0, key='pred_windspeed', step=0.1)
            st.slider("Visibilité (km)", 0.0, 50.0, key='pred_visibility', step=0.1)
        with col5:
            st.slider("Point de rosée (°C)", -20.0, 30.0, key='pred_tdewpoint', step=0.1)

        st.subheader("⚡ Données contextuelles (avancé)")
        col6, col7 = st.columns(2)
        with col6:
            st.number_input("Conso. moy. des 20 dernières min (Wh)", key='pred_app_roll_mean', step=1.0)
        with col7:
            st.number_input("Temp. ext. moy. des 50 dernières min (°C)", key='pred_t_out_roll_mean', step=0.1)
        
        # Le bouton de soumission appelle la fonction make_prediction
        st.form_submit_button('Lancer la Prédiction ✨', on_click=make_prediction)

    # --- 4. Affichage des résultats ---
    # On affiche les résultats seulement s'ils existent dans l'état de la session.
    if st.session_state.prediction_result is not None:
        st.markdown("---")
        st.subheader("📊 Résultat de la Prédiction")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric(
                label="Consommation d'énergie prédite (pour 10 min)", 
                value=f"{st.session_state.prediction_result:.2f} Wh"
            )
        
        with col_result2:
            delta = st.session_state.prediction_result - st.session_state.avg_for_hour_result
            st.metric(
                label=f"Comparaison avec la moyenne pour {st.session_state.hour_of_prediction_result}h", 
                value=f"{st.session_state.avg_for_hour_result:.2f} Wh", 
                delta=f"{delta:.2f} Wh", 
                delta_color="inverse"
            )
        
        st.markdown("---")
        st.button("🔄 Prédire une nouvelle valeur", on_click=reset_form, help="Efface le formulaire et les résultats pour une nouvelle prédiction.")

# 4. Clustering & Profils
elif menu == "🧩 Clustering et profils":
    st.markdown('<h1 class="page-header">🧩 Clustering et profils de consommation</h1>', unsafe_allow_html=True)
    clustering_features = [
        'hour', 'day_of_week', 'weekend', 'period_code',
        'temp_mean', 'hum_mean', 'confort_index',
        'T_out', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint'
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[clustering_features])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    df['cluster'] = clusters
    
    st.subheader("Projection PCA + KMeans (3 clusters)")
    fig8 = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=clusters.astype(str), labels={'x':'PC1','y':'PC2','color':'Cluster'})
    st.plotly_chart(fig8, use_container_width=True)
    
    st.subheader("Profils moyens par cluster")
    cluster_profiles = df.groupby('cluster')[clustering_features + ['Appliances']].mean().round(2)
    st.dataframe(cluster_profiles)

# 5. Détection d'anomalies
elif menu == "🚨 Détection d'anomalies":
    st.markdown('<h1 class="page-header">🚨 Détection d\'Anomalies</h1>', unsafe_allow_html=True)
    features_anomaly = [
        'Appliances', 'lights', 'hour', 'day_of_week', 'weekend', 'period_code',
        'temp_mean', 'hum_mean', 'confort_index', 'T_out', 'RH_out', 
        'Windspeed', 'Visibility', 'Tdewpoint'
    ]
    scaler = StandardScaler()
    X_scaled_anomaly = scaler.fit_transform(df[features_anomaly])
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X_scaled_anomaly)
    df['anomaly'] = pd.Series(anomaly_labels).map({1: 'normal', -1: 'anomalie'})
    
    st.subheader("Visualisation des anomalies détectées")
    fig9 = px.scatter(x=df.index, y=df['Appliances'], color=df['anomaly'],
                     color_discrete_map={'normal':'#636EFA','anomalie':'#EF553B'},
                     labels={'x':'Date','y':'Consommation (Wh)','color':'Type'})
    st.plotly_chart(fig9, use_container_width=True)
    st.write(f"Nombre d'anomalies détectées : {sum(df['anomaly']=='anomalie')}")

# 6. Optimisation (Q-learning)
elif menu == "🤖 Optimisation (Q-learning)":
    st.markdown('<h1 class="page-header">🤖 Optimisation par Q-Learning</h1>', unsafe_allow_html=True)
    
    # Recalculer les clusters pour cette section
    clustering_features_rl = [
        'hour', 'day_of_week', 'weekend', 'period_code',
        'temp_mean', 'hum_mean', 'confort_index', 'T_out', 'RH_out', 
        'Windspeed', 'Visibility', 'Tdewpoint'
    ]
    scaler_rl = StandardScaler()
    X_scaled_rl = scaler_rl.fit_transform(df[clustering_features_rl])
    pca_rl = PCA(n_components=2)
    X_pca_rl = pca_rl.fit_transform(X_scaled_rl)
    kmeans_rl = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans_rl.fit_predict(X_pca_rl)
    
    df_sim = df.copy()
    df_sim['day_type'] = df_sim['day_of_week'].apply(lambda x: 0 if x < 5 else 1)
    
    def simulate_consumption(row, action):
        base = row['Appliances']
        if action == 0: return base, 0
        elif action == 1: return base * 0.8, 0.5
        elif action == 2: return base * 0.5, 0.8
        else: return base, 1
        
    alpha, gamma, epsilon = 0.1, 0.9, 0.2
    q_table = defaultdict(lambda: [0.0, 0.0, 0.0])
    history = []
    rewards = []
    samples = df_sim.sample(2000, random_state=42).reset_index(drop=True)
    
    for i, row in samples.iterrows():
        state = (int(row['hour']), int(row['day_type']), int(row['cluster']))
        action = random.choice([0, 1, 2]) if random.random() < epsilon else np.argmax(q_table[state])
        simulated_conso, discomfort = simulate_consumption(row, action)
        reward = -simulated_conso - 10 * discomfort
        next_max = max(q_table[state])
        q_table[state][action] += alpha * (reward + gamma * next_max - q_table[state][action])
        history.append({'action': action})
        rewards.append(reward)
        
    hist_df = pd.DataFrame(history)
    st.subheader("Distribution des actions choisies par l'agent")
    fig10 = px.histogram(hist_df, x='action', nbins=3, labels={'action':'Action'},
                        category_orders={'action':[0,1,2]})
    st.plotly_chart(fig10, use_container_width=True)
    
    st.subheader("Évolution moyenne de la récompense (Q-learning)")
    window = 200
    if len(rewards) >= window:
        rolling = pd.Series(rewards).rolling(window).mean()
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(x=list(range(len(rolling))), y=rolling, mode='lines', name='reward'))
        fig_reward.update_layout(
            title=f"Évolution moyenne de la récompense (fenêtre glissante = {window})",
            xaxis_title="step",
            yaxis_title="reward",
            legend=dict(x=0.85, y=1.05),
            template="plotly_white"
        )
        st.plotly_chart(fig_reward, use_container_width=True)
    else:
        st.info("Pas assez d'itérations pour afficher la courbe de récompense.")
    
    st.subheader("Politique optimale par heure et cluster")
    policy = [{'hour': s[0], 'cluster': s[2], 'best_action': np.argmax(q_table[s])} for s in q_table.keys()]
    policy_df = pd.DataFrame(policy)
    if not policy_df.empty:
        pivot = policy_df.groupby(['cluster', 'hour'])['best_action'].agg(lambda x: pd.Series.mode(x)[0]).unstack()
        st.dataframe(pivot)
    st.markdown("**Légende des actions :** 0 = Ne rien faire, 1 = Réduire conso, 2 = Reporter usage")

# 7. Conclusion
elif menu == "✅ Conclusion":
    st.markdown('<h1 class="page-header">✅ Conclusion & Recommandations</h1>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="conclusion-box">
            <h3>Principaux Résultats</h3>
            <ul>
                <li>Le système permet de comprendre, prédire et optimiser la consommation énergétique.</li>
                <li>Les modèles supervisés atteignent un R² > 0.7, ce qui est très satisfaisant.</li>
                <li>Le clustering révèle des profils d'usage distincts.</li>
                <li>L'apprentissage par renforcement propose des stratégies d'optimisation concrètes.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="conclusion-box">
            <h3>Perspectives Futures</h3>
            <ul>
                <li>Intégrer d'autres sources de données (ex: production solaire, tarifs dynamiques).</li>
                <li>Déployer le système en temps réel avec des alertes personnalisées.</li>
                <li>Affiner les modèles avec des techniques d'optimisation plus poussées.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )