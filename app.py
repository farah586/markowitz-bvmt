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
    layout="wide"
)

st.title("📊 Tableau de bord Markowitz BVMT")
st.markdown("### Optimisation de portefeuille - Analyse financière avancée")

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
        
        # Trouver les colonnes
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
            st.error(f"Colonnes non trouvées dans {year}")
            return None
        
        df = df[[date_col, societe_col, close_col]].copy()
        df.columns = ["Date", "Societe", "Close"]
        
        # Nettoyage
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Date"])
        
        df["Close"] = df["Close"].astype(str).str.replace(",", ".", regex=False)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        df = df[df["Close"] > 0]
        
        # Filtrer par année
        df["Annee"] = df["Date"].dt.year
        df = df[df["Annee"] == year]
        
        if df.empty:
            return None
        
        # Filtrer les banques
        df["Societe"] = df["Societe"].astype(str).str.strip()
        mask = df["Societe"].str.upper().isin([b.upper() for b in BANQUES])
        df = df[mask]
        
        return df
    
    except Exception as e:
        st.error(f"Erreur: {e}")
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

def clean_dataframe(df):
    """Nettoie un DataFrame en remplaçant les valeurs infinies/NaN"""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def calculate_metrics(returns, rf):
    """Calcule les métriques financières"""
    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    
    # Éviter division par zéro
    volatility = volatility.replace(0, np.nan)
    sharpe = (mean_returns - rf) / volatility
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
    
    return mean_returns, volatility, sharpe


def optimize_portfolio(mean_returns, cov_matrix, rf):
    """Optimisation du portefeuille"""
    n = len(mean_returns)
    init = np.ones(n) / n
    
    def port_return(w):
        return np.sum(w * mean_returns)
    
    def port_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    def neg_sharpe(w):
        vol = port_vol(w)
        if vol < 0.0001 or np.isnan(vol):
            return 999
        ret = port_return(w) - rf
        if np.isnan(ret):
            return 999
        return -ret / vol
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    try:
        result = minimize(
            neg_sharpe, init, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 500}
        )
        weights = result.x if result.success else init
    except:
        weights = init
    
    # Nettoyer les poids
    weights = np.nan_to_num(weights, nan=0)
    weights = weights / weights.sum() if weights.sum() > 0 else init
    
    ret_opt = port_return(weights)
    vol_opt = port_vol(weights)
    sharpe_opt = (ret_opt - rf) / vol_opt if vol_opt > 0 and not np.isnan(vol_opt) else 0
    
    return weights, ret_opt, vol_opt, sharpe_opt


# ==============================
# CHARGEMENT DES DONNÉES
# ==============================

BASE_DIR = Path(__file__).parent
YEARS = [2023, 2024, 2025]

st.sidebar.header("⚙️ Configuration")

selected_year = st.sidebar.selectbox(
    "📅 Année à analyser",
    YEARS,
    index=len(YEARS)-1
)

# Trouver le fichier
file_path = None
for f in BASE_DIR.iterdir():
    if f.is_file() and f.suffix.lower() in ['.xlsx', '.xls']:
        if str(selected_year) in f.stem:
            file_path = f
            break

if file_path is None:
    st.error(f"❌ Fichier pour {selected_year} non trouvé!")
    st.stop()

# Chargement
with st.spinner(f"📂 Chargement {selected_year}..."):
    data = load_excel_file(file_path, selected_year)
    
    if data is None or data.empty:
        st.error(f"❌ Aucune donnée pour {selected_year}")
        st.stop()
    
    prices = prepare_prices(data)
    
    if prices is None or prices.empty:
        st.error("❌ Erreur préparation prix")
        st.stop()

st.sidebar.success(f"✅ {len(prices.columns)} banques chargées")

# Sélection des banques
selected_banks = st.sidebar.multiselect(
    "🏦 Banques à analyser",
    options=sorted(prices.columns.tolist()),
    default=sorted(prices.columns.tolist())[:min(8, len(prices.columns))]
)

if len(selected_banks) < 2:
    st.warning("Sélectionnez au moins 2 banques")
    if len(prices.columns) >= 2:
        selected_banks = sorted(prices.columns.tolist())[:2]
    else:
        st.stop()

# Paramètres
rf = st.sidebar.number_input("Taux sans risque (%)", value=7.5, step=0.5) / 100
capital = st.sidebar.number_input("Capital (TND)", value=10000, step=5000)

# ==============================
# CALCULS
# ==============================

with st.spinner("🔄 Calculs..."):
    selected_prices = prices[selected_banks].dropna(how="all").ffill()
    returns = selected_prices.pct_change().dropna()
    
    if returns.empty or len(returns) < 10:
        st.error("Pas assez de données")
        st.stop()
    
    # Métriques
    mean_returns, volatility, sharpe_individual = calculate_metrics(returns, rf)
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()
    
    # Nettoyer les matrices
    cov_matrix = cov_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
    corr_matrix = corr_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # Drawdown (éviter division par zéro)
    cummax = selected_prices.cummax()
    cummax = cummax.replace(0, np.nan)
    drawdown = (selected_prices - cummax) / cummax
    drawdown = drawdown.fillna(0)
    max_drawdown = drawdown.min()
    
    # VaR
    var_95 = returns.quantile(0.05) * np.sqrt(252)
    var_95 = var_95.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Beta
    market_returns = returns.mean(axis=1)
    betas = {}
    for bank in returns.columns:
        try:
            cov = np.cov(returns[bank], market_returns)[0, 1] if len(market_returns) > 1 else 0
            var = np.var(market_returns) if len(market_returns) > 1 else 1
            beta_val = cov / var if var != 0 else 1
            betas[bank] = beta_val if not np.isnan(beta_val) else 1
        except:
            betas[bank] = 1
    betas = pd.Series(betas)
    
    # Optimisation
    weights_opt, ret_opt, vol_opt, sharpe_opt = optimize_portfolio(
        mean_returns.fillna(0).values, 
        cov_matrix.values, 
        rf
    )
    
    # Portfolio VaR
    portfolio_returns = returns.dot(weights_opt)
    portfolio_var_95 = portfolio_returns.quantile(0.05) * np.sqrt(252)
    
    # DataFrame des poids
    weights_df = pd.DataFrame({
        "Banque": selected_banks,
        "Poids optimal": weights_opt,
        "Montant (TND)": weights_opt * capital
    })
    weights_df = weights_df[weights_df["Poids optimal"] > 0.001]
    weights_df = weights_df.sort_values("Poids optimal", ascending=False).reset_index(drop=True)
    
    # DataFrame des métriques (nettoyé)
    metrics_df = pd.DataFrame({
        "Banque": selected_banks,
        "Rentabilité annualisée": mean_returns.values,
        "Volatilité annualisée": volatility.values,
        "Ratio de Sharpe": sharpe_individual.values,
        "Drawdown max (%)": max_drawdown.values * 100,
        "VaR 95%": var_95.values * 100,
        "Beta": betas.values
    })
    
    # Remplacer les valeurs problématiques
    metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan)
    metrics_df = metrics_df.fillna(0)
    
    # Supprimer les lignes avec des valeurs aberrantes
    metrics_df = metrics_df[metrics_df["Rentabilité annualisée"].abs() < 5]
    metrics_df = metrics_df[metrics_df["Volatilité annualisée"].abs() < 5]
    metrics_df = metrics_df[metrics_df["Ratio de Sharpe"].abs() < 10]
    
    # Classement
    if len(metrics_df) > 0:
        metrics_df["Score"] = (
            metrics_df["Ratio de Sharpe"].rank(ascending=False, method='dense') +
            metrics_df["Rentabilité annualisée"].rank(ascending=False, method='dense') +
            metrics_df["Volatilité annualisée"].rank(ascending=True, method='dense') +
            metrics_df["Drawdown max (%)"].rank(ascending=True, method='dense') +
            metrics_df["VaR 95%"].rank(ascending=True, method='dense')
        )
        metrics_df = metrics_df.sort_values("Score").reset_index(drop=True)
        best_bank = metrics_df.iloc[0]["Banque"] if len(metrics_df) > 0 else "N/A"
    else:
        best_bank = "N/A"

# ==============================
# AFFICHAGE
# ==============================

# KPIs
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("🏦 Banques", len(selected_banks))
with col2:
    st.metric("📈 Rentabilité", f"{max(0, ret_opt):.2%}")
with col3:
    st.metric("⚠️ Risque", f"{vol_opt:.2%}")
with col4:
    st.metric("🎯 Sharpe", f"{sharpe_opt:.3f}")
with col5:
    st.metric("🏆 Meilleure", best_bank[:15] if best_bank != "N/A" else "N/A")

st.markdown("---")

# ONGLETS
tab1, tab2, tab3, tab4 = st.tabs(["📈 Vue générale", "📊 Métriques", "🎯 Optimisation", "📥 Export"])

with tab1:
    # Évolution des cours
    if not selected_prices.empty:
        st.subheader("📈 Évolution des cours")
        fig_prices = px.line(selected_prices, title=f"Cours - {selected_year}")
        fig_prices.update_layout(height=450)
        st.plotly_chart(fig_prices, use_container_width=True)
    
    # Rendements cumulés
    if not cumulative_returns.empty:
        st.subheader("📊 Performance cumulée")
        fig_cum = px.line(cumulative_returns, title="Rendements cumulés")
        fig_cum.update_yaxes(tickformat=".0%")
        fig_cum.update_layout(height=400)
        st.plotly_chart(fig_cum, use_container_width=True)
    
    # Carte rendement/risque (version corrigée)
    if not metrics_df.empty and len(metrics_df) > 1:
        st.subheader("🗺️ Carte rendement / risque")
        
        # Filtrer les données valides
        plot_df = metrics_df.dropna(subset=["Volatilité annualisée", "Rentabilité annualisée", "Ratio de Sharpe"])
        
        if not plot_df.empty and len(plot_df) > 1:
            fig_scatter = px.scatter(
                plot_df,
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
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Données insuffisantes pour la carte rendement/risque")

with tab2:
    st.subheader("📊 Métriques par banque")
    if not metrics_df.empty:
        st.dataframe(
            metrics_df.style.format({
                "Rentabilité annualisée": "{:.2%}",
                "Volatilité annualisée": "{:.2%}",
                "Ratio de Sharpe": "{:.3f}",
                "Drawdown max (%)": "{:.1f}%",
                "VaR 95%": "{:.1f}%",
                "Beta": "{:.2f}"
            }),
            use_container_width=True,
            height=400
        )
        
        # Graphiques
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig_ret = px.bar(metrics_df, x="Banque", y="Rentabilité annualisée", title="Rentabilité")
            fig_ret.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_ret, use_container_width=True)
        
        with col_b:
            fig_sharpe = px.bar(metrics_df, x="Banque", y="Ratio de Sharpe", title="Ratio de Sharpe")
            st.plotly_chart(fig_sharpe, use_container_width=True)

with tab3:
    st.subheader("🎯 Portefeuille optimal")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Rentabilité", f"{max(0, ret_opt):.2%}")
    with col_b:
        st.metric("Risque", f"{vol_opt:.2%}")
    with col_c:
        st.metric("Sharpe", f"{sharpe_opt:.3f}")
    
    st.markdown("---")
    
    if not weights_df.empty:
        col_pie, col_table = st.columns([1, 1.5])
        
        with col_pie:
            fig_pie = px.pie(weights_df.head(8), names="Banque", values="Poids optimal", title="Répartition")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_table:
            st.dataframe(
                weights_df.style.format({"Poids optimal": "{:.2%}", "Montant (TND)": "{:,.2f}"}),
                use_container_width=True
            )

with tab4:
    st.subheader("📥 Export")
    
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            selected_prices.to_excel(writer, sheet_name="Prix")
            returns.to_excel(writer, sheet_name="Rendements")
            metrics_df.to_excel(writer, sheet_name="Metriques", index=False)
            weights_df.to_excel(writer, sheet_name="Allocation", index=False)
        
        st.download_button(
            label="📥 Télécharger Excel",
            data=output.getvalue(),
            file_name=f"markowitz_{selected_year}.xlsx",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Erreur: {e}")

st.markdown("---")
st.caption("⚠️ Analyse éducative - Ne constitue pas un conseil en investissement")
