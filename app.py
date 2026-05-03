import io
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy import stats
import streamlit as st

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Markowitz BVMT - Optimisation de Portefeuille",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Tableau de bord Markowitz BVMT")
st.markdown("### Optimisation de portefeuille - Analyse financière avancée")
st.markdown("---")

# ==============================
# LISTE DES BANQUES TUNISIENNES
# ==============================

BANQUES = [
    "BIAT", "ATB", "STB", "BT", "AMEN BANK", "UIB", "UBCI", "BH",
    "BNA", "ATTIJARI BANK", "BH BANK", "BTE", "WIFACK INT BANK"
]

# ==============================
# FONCTIONS DE CHARGEMENT
# ==============================

@st.cache_data
def load_excel_file(file_path, year):
    """Charge un fichier Excel et filtre les banques"""
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        
        st.sidebar.write(f"📋 Colonnes trouvées dans {year}:", list(df.columns))
        
        # Trouver la colonne des noms de sociétés (peut être VALEUR ou VALUEUR)
        societe_col = None
        date_col = None
        close_col = None
        
        for col in df.columns:
            if col.upper() in ['VALEUR', 'VALUEUR', 'SOCIETE', 'NOM']:
                societe_col = col
            elif col.upper() in ['SEANCE', 'DATE']:
                date_col = col
            elif col.upper() in ['CLOTURE', 'CLOSE', 'PRIX']:
                close_col = col
        
        if societe_col is None or date_col is None or close_col is None:
            st.error(f"Colonnes requises non trouvées dans {year}")
            st.write(f"Colonnes disponibles: {list(df.columns)}")
            return None
        
        # Sélection des colonnes
        df = df[[date_col, societe_col, close_col]].copy()
        df.columns = ["Date", "Societe", "Close"]
        
        # Nettoyage des dates
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Date"])
        
        # Nettoyage des prix
        df["Close"] = df["Close"].astype(str).str.replace(",", ".", regex=False)
        df["Close"] = df["Close"].astype(str).str.replace(" ", "", regex=False)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        df = df[df["Close"] > 0]
        
        # Filtrer par année
        df["Annee"] = df["Date"].dt.year
        df = df[df["Annee"] == year]
        
        if df.empty:
            st.warning(f"Aucune donnée pour l'année {year}")
            return None
        
        # Nettoyer les noms des sociétés
        df["Societe"] = df["Societe"].astype(str).str.strip()
        
        # Filtrer les banques (correspondance approximative)
        banques_trouvees = []
        for bank in BANQUES:
            mask = df["Societe"].str.upper().str.contains(bank.upper(), na=False)
            if mask.any():
                banques_trouvees.append(bank)
                df.loc[mask, "Societe"] = bank  # Uniformiser le nom
        
        if not banques_trouvees:
            st.warning(f"Aucune banque trouvée dans {year}")
            st.write("Premières sociétés disponibles:")
            st.write(df["Societe"].unique()[:20])
            return None
        
        # Garder seulement les banques
        df = df[df["Societe"].isin(banques_trouvees)]
        
        st.sidebar.success(f"✅ {year}: {len(df['Societe'].unique())} banques trouvées")
        
        return df
    
    except Exception as e:
        st.error(f"Erreur chargement {year}: {str(e)}")
        return None


@st.cache_data
def prepare_prices(data):
    """Prépare la matrice des prix"""
    if data is None or data.empty:
        return None
    
    prices = data.pivot_table(
        index="Date",
        columns="Societe",
        values="Close",
        aggfunc="first"
    ).sort_index().ffill()
    
    return prices


# ==============================
# FONCTIONS D'OPTIMISATION
# ==============================

def calculate_metrics(returns, rf):
    """Calcule les métriques financières"""
    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe = (mean_returns - rf) / volatility
    return mean_returns, volatility, sharpe


def optimize_portfolio(mean_returns, cov_matrix, rf):
    """Optimisation du portefeuille (Sharpe max)"""
    n = len(mean_returns)
    init = np.ones(n) / n
    
    def port_return(w):
        return np.sum(w * mean_returns)
    
    def port_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    def neg_sharpe(w):
        vol = port_vol(w)
        if vol < 0.0001:
            return 999
        return -(port_return(w) - rf) / vol
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    try:
        result = minimize(
            neg_sharpe, init, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        weights = result.x if result.success else init
    except:
        weights = init
    
    ret_opt = port_return(weights)
    vol_opt = port_vol(weights)
    sharpe_opt = (ret_opt - rf) / vol_opt if vol_opt > 0 else 0
    
    return weights, ret_opt, vol_opt, sharpe_opt


def calculate_var(returns, confidence=0.95):
    """Calcule la Value at Risk"""
    return returns.quantile(1 - confidence) * np.sqrt(252)


def calculate_cvar(returns, confidence=0.95):
    """Calcule la Conditional Value at Risk (Expected Shortfall)"""
    var = returns.quantile(1 - confidence)
    return returns[returns <= var].mean() * np.sqrt(252)


def calculate_beta(returns, market_returns):
    """Calcule le beta par rapport au marché"""
    if len(market_returns) > 1:
        cov = np.cov(returns, market_returns)[0, 1]
        var = np.var(market_returns)
        return cov / var if var != 0 else 1
    return 1


# ==============================
# CHARGEMENT DES DONNÉES
# ==============================

BASE_DIR = Path(__file__).parent
YEARS = [2023, 2024, 2025]

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Sélection de l'année
selected_year = st.sidebar.selectbox(
    "📅 Année à analyser",
    YEARS,
    index=len(YEARS)-1,
    help="Sélectionnez l'année pour l'analyse"
)

# Recherche du fichier
file_path = None
for f in BASE_DIR.iterdir():
    if f.is_file() and f.suffix.lower() in ['.xlsx', '.xls']:
        if str(selected_year) in f.stem:
            file_path = f
            break

if file_path is None:
    st.error(f"❌ Fichier pour {selected_year} non trouvé!")
    st.info(f"Fichiers trouvés: {[f.name for f in BASE_DIR.iterdir() if f.suffix in ['.xlsx', '.xls']]}")
    st.stop()

st.sidebar.info(f"📁 Fichier: {file_path.name}")

# Chargement
with st.spinner(f"📂 Chargement des données {selected_year}..."):
    data = load_excel_file(file_path, selected_year)
    
    if data is None or data.empty:
        st.error(f"❌ Aucune donnée valide pour {selected_year}")
        st.stop()
    
    prices = prepare_prices(data)
    
    if prices is None or prices.empty:
        st.error("❌ Erreur lors de la préparation des prix")
        st.stop()

# Affichage des banques disponibles
st.sidebar.success(f"✅ {len(prices.columns)} banques chargées")

if len(prices.columns) > 0:
    with st.sidebar.expander("🏦 Banques disponibles"):
        for bank in sorted(prices.columns):
            st.write(f"• {bank}")
else:
    st.error("❌ Aucune banque trouvée! Vérifiez le nom des colonnes dans votre fichier Excel.")
    st.write("Le fichier doit contenir une colonne 'SEANCE' (date), 'VALEUR' ou 'VALUEUR' (nom), et 'CLOTURE' (prix)")
    st.stop()

# Sélection des banques
selected_banks = st.sidebar.multiselect(
    "🏦 Banques à analyser",
    options=sorted(prices.columns.tolist()),
    default=sorted(prices.columns.tolist()),
    help="Sélectionnez les banques pour l'analyse"
)

if len(selected_banks) < 2:
    st.warning("⚠️ Veuillez sélectionner au moins 2 banques")
    if len(prices.columns) >= 2:
        selected_banks = sorted(prices.columns.tolist())[:2]
    else:
        st.stop()

# Paramètres financiers
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Paramètres financiers")

rf = st.sidebar.number_input(
    "Taux sans risque (%)",
    value=7.5,
    step=0.5,
    help="Taux des obligations d'État tunisiennes"
) / 100

capital = st.sidebar.number_input(
    "Capital à investir (TND)",
    value=10000,
    step=5000,
    help="Montant total à investir"
)

# ==============================
# CALCULS PRINCIPAUX
# ==============================

with st.spinner("🔄 Calculs en cours..."):
    # Filtrage des prix
    selected_prices = prices[selected_banks].dropna(how="all").ffill()
    
    # Calcul des rendements
    returns = selected_prices.pct_change().dropna()
    
    if returns.empty or len(returns) < 10:
        st.error("❌ Pas assez de données pour l'analyse")
        st.stop()
    
    # Métriques annualisées
    mean_returns, volatility, sharpe_individual = calculate_metrics(returns, rf)
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()
    
    # Rendements cumulés
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # Drawdown
    cummax = selected_prices.cummax()
    drawdown = (selected_prices - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # VaR et CVaR
    var_90 = calculate_var(returns, 0.90)
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    cvar_95 = calculate_cvar(returns, 0.95)
    
    # Beta marché
    market_returns = returns.mean(axis=1)
    betas = pd.Series({
        bank: calculate_beta(returns[bank], market_returns)
        for bank in returns.columns
    })
    
    # Optimisation du portefeuille
    weights_opt, ret_opt, vol_opt, sharpe_opt = optimize_portfolio(
        mean_returns, cov_matrix, rf
    )
    
    # Portfolio VaR
    portfolio_returns = returns.dot(weights_opt)
    portfolio_var_95 = calculate_var(portfolio_returns, 0.95)
    
    # DataFrame des poids
    weights_df = pd.DataFrame({
        "Banque": selected_banks,
        "Poids optimal": weights_opt,
        "Montant (TND)": weights_opt * capital
    })
    weights_df = weights_df[weights_df["Poids optimal"] > 0.001]
    weights_df = weights_df.sort_values("Poids optimal", ascending=False).reset_index(drop=True)
    
    # DataFrame des métriques
    metrics_df = pd.DataFrame({
        "Banque": selected_banks,
        "Rentabilité annualisée": mean_returns.values,
        "Volatilité annualisée": volatility.values,
        "Ratio de Sharpe": sharpe_individual.values,
        "Drawdown max (%)": max_drawdown.values * 100,
        "VaR 95%": var_95.values * 100,
        "Beta": betas.values
    })
    
    # Classement des banques
    metrics_df["Score"] = (
        metrics_df["Ratio de Sharpe"].rank(ascending=False) +
        metrics_df["Rentabilité annualisée"].rank(ascending=False) +
        metrics_df["Volatilité annualisée"].rank(ascending=True) +
        metrics_df["Drawdown max (%)"].rank(ascending=True) +
        metrics_df["VaR 95%"].rank(ascending=True)
    )
    metrics_df = metrics_df.sort_values("Score").reset_index(drop=True)
    best_bank = metrics_df.iloc[0]["Banque"] if len(metrics_df) > 0 else "N/A"
    
    # Recommandations
    def get_recommendation(row):
        median_vol = metrics_df["Volatilité annualisée"].median()
        if row["Ratio de Sharpe"] > 1 and row["Volatilité annualisée"] < median_vol:
            return "🟢 Très attractive"
        elif row["Ratio de Sharpe"] > 0.5:
            return "🟡 Intéressante"
        elif row["Volatilité annualisée"] > median_vol:
            return "🔴 Risque élevé"
        else:
            return "⚪ À surveiller"
    
    metrics_df["Recommandation"] = metrics_df.apply(get_recommendation, axis=1)

# ==============================
# INTERFACE PRINCIPALE
# ==============================

st.header(f"📊 Analyse {selected_year} - Portefeuille bancaire optimal")

# KPIS
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("🏦 Banques", len(selected_banks))
with col2:
    st.metric("📈 Rentabilité", f"{ret_opt:.2%}")
with col3:
    st.metric("⚠️ Risque", f"{vol_opt:.2%}")
with col4:
    st.metric("🎯 Sharpe", f"{sharpe_opt:.3f}")
with col5:
    st.metric("🏆 Meilleure", best_bank[:15])

st.markdown("---")

# ========== TAB 1: VUE GÉNÉRALE ==========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Vue générale",
    "📊 Métriques",
    "⚠️ Risques",
    "🎯 Optimisation",
    "🔗 Corrélations",
    "📥 Export"
])

with tab1:
    # Évolution des cours
    if not selected_prices.empty:
        st.subheader("📈 Évolution des cours")
        fig_prices = px.line(
            selected_prices,
            title=f"Cours de clôture - {selected_year}",
            labels={"value": "Prix (TND)", "variable": "Banque"}
        )
        fig_prices.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig_prices, use_container_width=True)
    
    # Rendements cumulés
    if not cumulative_returns.empty:
        st.subheader("📊 Performance cumulée")
        fig_cum = px.line(
            cumulative_returns,
            title="Rendements cumulés",
            labels={"value": "Rendement", "variable": "Banque"}
        )
        fig_cum.update_yaxes(tickformat=".0%")
        fig_cum.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig_cum, use_container_width=True)
    
    # Carte rendement/risque
    if not metrics_df.empty:
        st.subheader("🗺️ Carte rendement / risque")
        fig_scatter = px.scatter(
            metrics_df,
            x="Volatilité annualisée",
            y="Rentabilité annualisée",
            size="Ratio de Sharpe",
            color="Ratio de Sharpe",
            text="Banque",
            title="Positionnement des banques",
            labels={"Volatilité annualisée": "Risque", "Rentabilité annualisée": "Rendement"}
        )
        fig_scatter.update_xaxes(tickformat=".0%")
        fig_scatter.update_yaxes(tickformat=".0%")
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("📊 Métriques détaillées par banque")
    st.dataframe(
        metrics_df.style.format({
            "Rentabilité annualisée": "{:.2%}",
            "Volatilité annualisée": "{:.2%}",
            "Ratio de Sharpe": "{:.3f}",
            "Drawdown max (%)": "{:.1f}%",
            "VaR 95%": "{:.1f}%",
            "Beta": "{:.2f}",
            "Score": "{:.0f}"
        }).background_gradient(subset=["Ratio de Sharpe"], cmap="RdYlGn"),
        use_container_width=True,
        height=400
    )
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_ret = px.bar(
            metrics_df,
            x="Banque",
            y="Rentabilité annualisée",
            title="Rentabilité par banque",
            color="Rentabilité annualisée",
            color_continuous_scale="Viridis"
        )
        fig_ret.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ret, use_container_width=True)
        
        fig_sharpe = px.bar(
            metrics_df,
            x="Banque",
            y="Ratio de Sharpe",
            title="Ratio de Sharpe",
            color="Ratio de Sharpe",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)
    
    with col_b:
        fig_vol = px.bar(
            metrics_df,
            x="Banque",
            y="Volatilité annualisée",
            title="Risque par banque",
            color="Volatilité annualisée",
            color_continuous_scale="OrRd"
        )
        fig_vol.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_vol, use_container_width=True)
        
        fig_beta = px.bar(
            metrics_df,
            x="Banque",
            y="Beta",
            title="Beta (risque systématique)",
            color="Beta",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_beta, use_container_width=True)

with tab3:
    if not drawdown.empty:
        st.subheader("📉 Drawdown (perte maximale)")
        fig_dd = px.area(
            drawdown,
            title="Drawdown des banques",
            labels={"value": "Perte (%)", "variable": "Banque"}
        )
        fig_dd.update_yaxes(tickformat=".0%")
        fig_dd.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig_dd, use_container_width=True)
    
    if not metrics_df.empty:
        st.subheader("📊 Value at Risk (VaR) à 95%")
        fig_var = px.bar(
            metrics_df,
            x="Banque",
            y="VaR 95%",
            title="Perte maximale attendue (95% de confiance)",
            color="VaR 95%",
            color_continuous_scale="Reds"
        )
        fig_var.update_yaxes(tickformat=".1f")
        st.plotly_chart(fig_var, use_container_width=True)

with tab4:
    st.subheader("🎯 Portefeuille optimal (Sharpe maximum)")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Rentabilité annualisée", f"{ret_opt:.2%}")
    with col_b:
        st.metric("Volatilité annualisée", f"{vol_opt:.2%}")
    with col_c:
        st.metric("Ratio de Sharpe", f"{sharpe_opt:.3f}")
    
    st.markdown("---")
    
    col_pie, col_table = st.columns([1, 1.5])
    
    with col_pie:
        if len(weights_df) > 0:
            fig_pie = px.pie(
                weights_df,
                names="Banque",
                values="Poids optimal",
                title="Répartition du portefeuille",
                hole=0.3
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=450)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_table:
        st.dataframe(
            weights_df.style.format({
                "Poids optimal": "{:.2%}",
                "Montant (TND)": "{:,.2f}"
            }).bar(subset=["Poids optimal"], color="#2ecc71"),
            use_container_width=True,
            height=400
        )
    
    if len(weights_df) > 0:
        fig_alloc = px.bar(
            weights_df,
            x="Banque",
            y="Poids optimal",
            title="Allocation par banque",
            color="Poids optimal",
            color_continuous_scale="Viridis",
            text_auto=".1%"
        )
        fig_alloc.update_yaxes(tickformat=".0%")
        fig_alloc.update_layout(height=450)
        st.plotly_chart(fig_alloc, use_container_width=True)

with tab5:
    if not corr_matrix.empty:
        st.subheader("🔗 Matrice de corrélation")
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Corrélations entre banques",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("🎨 Matrice de covariance annualisée")
        fig_cov = px.imshow(
            cov_matrix,
            text_auto=True,
            aspect="auto",
            title="Covariance annualisée",
            color_continuous_scale="Viridis"
        )
        fig_cov.update_layout(height=600)
        st.plotly_chart(fig_cov, use_container_width=True)

with tab6:
    st.subheader("📥 Export des résultats")
    st.info("Téléchargez un rapport Excel complet avec toutes les analyses")
    
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            selected_prices.to_excel(writer, sheet_name="1_Prix")
            returns.to_excel(writer, sheet_name="2_Rendements")
            cumulative_returns.to_excel(writer, sheet_name="3_Rendements_cumules")
            metrics_df.to_excel(writer, sheet_name="4_Metriques_banques", index=False)
            weights_df.to_excel(writer, sheet_name="5_Allocation_portefeuille", index=False)
            cov_matrix.to_excel(writer, sheet_name="6_Matrice_covariance")
            corr_matrix.to_excel(writer, sheet_name="7_Matrice_correlation")
            if not drawdown.empty:
                drawdown.to_excel(writer, sheet_name="8_Drawdown")
            
            portfolio_stats = pd.DataFrame({
                "Métrique": [
                    "Rentabilité annualisée", "Volatilité annualisée", "Ratio de Sharpe",
                    "VaR 95%", "Capital investi", "Nombre de banques",
                    "Taux sans risque", "Année analysée"
                ],
                "Valeur": [
                    f"{ret_opt:.2%}", f"{vol_opt:.2%}", f"{sharpe_opt:.3f}",
                    f"{portfolio_var_95:.2%}", f"{capital:,.0f} TND",
                    len(weights_df), f"{rf:.1%}", selected_year
                ]
            })
            portfolio_stats.to_excel(writer, sheet_name="9_Stats_portefeuille", index=False)
        
        st.download_button(
            label="📥 Télécharger le rapport Excel complet",
            data=output.getvalue(),
            file_name=f"markowitz_bvmt_{selected_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Erreur lors de la création du rapport: {e}")

# ==============================
# RECOMMANDATIONS FINALES
# ==============================

st.markdown("---")
st.subheader("🤖 Recommandations intelligentes")

col_rec1, col_rec2, col_rec3 = st.columns(3)

with col_rec1:
    st.info(f"🏆 **Meilleure banque**\n\n{best_bank}\n\n*Basé sur le score composite*")

with col_rec2:
    if sharpe_opt > 1:
        st.success(f"📈 **Excellent ratio de Sharpe**\n\n{sharpe_opt:.3f}\n\nLe portefeuille offre un excellent rendement ajusté au risque")
    elif sharpe_opt > 0.5:
        st.info(f"📊 **Bon ratio de Sharpe**\n\n{sharpe_opt:.3f}\n\nLe portefeuille offre un bon rendement ajusté au risque")
    else:
        st.warning(f"⚠️ **Ratio de Sharpe faible**\n\n{sharpe_opt:.3f}\n\nLe rendement ne compense pas suffisamment le risque")

with col_rec3:
    st.warning("⚠️ **Avertissement**\n\nCette analyse est basée sur des données historiques et ne constitue pas un conseil en investissement")

st.markdown("---")
st.caption(
    "📊 **Méthodologie:** Analyse Markowitz | Annualisation: 252 jours | "
    "VaR 95% historique | Données: BVMT"
)
