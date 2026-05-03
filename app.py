import io
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit as st

warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(page_title="Markowitz BVMT", layout="wide")

st.title("📊 Tableau de bord Markowitz BVMT")
st.markdown("### Optimisation de portefeuille - Analyse financière avancée")

# ==============================
# LISTE DES BANQUES
# ==============================

BANQUES = [
    "BIAT", "ATB", "STB", "BT", "AMEN BANK", "UIB", "UBCI", "BH",
    "BNA", "ATTIJARI BANK", "BH BANK", "BTE", "WIFACK INT BANK"
]

# ==============================
# CHARGEMENT
# ==============================

@st.cache_data
def load_data(file_path, year):
    """Charge et filtre les données"""
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        
        # Trouver les colonnes
        societe_col = None
        date_col = None
        close_col = None
        
        for col in df.columns:
            col_upper = col.upper()
            if col_upper in ['VALEUR', 'VALUEUR', 'SOCIETE', 'NOM']:
                societe_col = col
            elif col_upper in ['SEANCE', 'DATE']:
                date_col = col
            elif col_upper in ['CLOTURE', 'CLOSE', 'PRIX']:
                close_col = col
        
        if not all([societe_col, date_col, close_col]):
            return None
        
        df = df[[date_col, societe_col, close_col]].copy()
        df.columns = ["Date", "Societe", "Close"]
        
        # Nettoyage
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Date"])
        
        df["Close"] = df["Close"].astype(str).str.replace(",", ".")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        df = df[df["Close"] > 0]
        
        # Filtre année
        df["Annee"] = df["Date"].dt.year
        df = df[df["Annee"] == year]
        
        if df.empty:
            return None
        
        # Filtre banques
        df["Societe"] = df["Societe"].astype(str).str.strip()
        mask = df["Societe"].str.upper().isin([b.upper() for b in BANQUES])
        df = df[mask]
        
        return df
    
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None


@st.cache_data
def get_prices(data):
    """Crée la matrice des prix"""
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
# OPTIMISATION
# ==============================

def safe_optimize(returns, rf):
    """Optimisation robuste avec gestion d'erreurs"""
    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    cov_matrix = returns.cov() * 252
    
    # Remplacer les valeurs problématiques
    mean_returns = mean_returns.replace([np.inf, -np.inf], 0).fillna(0)
    volatility = volatility.replace([np.inf, -np.inf], 0).fillna(0)
    cov_matrix = cov_matrix.replace([np.inf, -np.inf], 0).fillna(0)
    
    n = len(mean_returns)
    init = np.ones(n) / n
    
    try:
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n))
        
        def neg_sharpe(w):
            ret = np.sum(w * mean_returns)
            vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            if vol < 0.0001:
                return 999
            return -(ret - rf) / vol
        
        result = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 200})
        weights = result.x if result.success else init
        
    except:
        weights = init
    
    # Normaliser
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum() if weights.sum() > 0 else init
    
    ret_opt = np.sum(weights * mean_returns)
    vol_opt = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_opt = (ret_opt - rf) / vol_opt if vol_opt > 0 else 0
    
    return weights, ret_opt, vol_opt, sharpe_opt, mean_returns, volatility, cov_matrix


# ==============================
# MAIN
# ==============================

BASE_DIR = Path(__file__).parent
YEARS = [2023, 2024, 2025]

st.sidebar.header("⚙️ Configuration")

selected_year = st.sidebar.selectbox("📅 Année", YEARS, index=len(YEARS)-1)

# Trouver le fichier
file_path = None
for f in BASE_DIR.iterdir():
    if f.is_file() and f.suffix in ['.xlsx', '.xls']:
        if str(selected_year) in f.stem:
            file_path = f
            break

if not file_path:
    st.error(f"Fichier {selected_year} non trouvé")
    st.stop()

# Chargement
with st.spinner(f"Chargement {selected_year}..."):
    data = load_data(file_path, selected_year)
    if data is None or data.empty:
        st.error("Aucune donnée chargée")
        st.stop()
    
    prices = get_prices(data)
    if prices is None or prices.empty:
        st.error("Erreur préparation prix")
        st.stop()

st.sidebar.success(f"✅ {len(prices.columns)} banques")

# Sélection
selected_banks = st.sidebar.multiselect(
    "Banques à analyser",
    options=sorted(prices.columns),
    default=sorted(prices.columns)[:min(6, len(prices.columns))]
)

if len(selected_banks) < 2:
    selected_banks = sorted(prices.columns)[:2]

rf = st.sidebar.number_input("Taux sans risque (%)", value=8.0, step=0.5) / 100
capital = st.sidebar.number_input("Capital (TND)", value=10000, step=5000)

# ==============================
# CALCULS
# ==============================

with st.spinner("Calculs..."):
    selected_prices = prices[selected_banks].dropna(how="all").ffill()
    returns = selected_prices.pct_change().dropna()
    
    if returns.empty or len(returns) < 5:
        st.error("Pas assez de données")
        st.stop()
    
    # Optimisation
    weights, ret_opt, vol_opt, sharpe_opt, mean_rets, vols, cov_mat = safe_optimize(returns, rf)
    
    # Portfolio VaR
    port_returns = returns.dot(weights)
    var_95 = port_returns.quantile(0.05) * np.sqrt(252)
    
    # Résultats
    weights_df = pd.DataFrame({
        "Banque": selected_banks,
        "Poids": weights,
        "Montant (TND)": weights * capital
    })
    weights_df = weights_df[weights_df["Poids"] > 0.001].sort_values("Poids", ascending=False).reset_index(drop=True)
    
    # Métriques individuelles
    metrics_data = []
    for i, bank in enumerate(selected_banks):
        metrics_data.append({
            "Banque": bank,
            "Rendement": mean_rets.iloc[i] if hasattr(mean_rets, 'iloc') else mean_rets[i],
            "Risque": vols.iloc[i] if hasattr(vols, 'iloc') else vols[i],
            "Sharpe": (mean_rets.iloc[i] - rf) / vols.iloc[i] if vols.iloc[i] > 0 else 0
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Best bank
    if len(metrics_df) > 0:
        best_idx = metrics_df["Sharpe"].idxmax()
        best_bank = metrics_df.iloc[best_idx]["Banque"]
    else:
        best_bank = "N/A"
    
    # Cumulative returns
    cum_returns = (1 + returns).cumprod() - 1
    
    # Drawdown
    cummax = selected_prices.cummax()
    drawdown = (selected_prices - cummax) / cummax
    drawdown = drawdown.fillna(0)

# ==============================
# AFFICHAGE
# ==============================

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("🏦 Banques", len(selected_banks))
col2.metric("📈 Rendement optimal", f"{max(-0.5, min(0.5, ret_opt)):.2%}")
col3.metric("⚠️ Risque optimal", f"{vol_opt:.2%}")
col4.metric("🎯 Sharpe optimal", f"{sharpe_opt:.3f}")

st.markdown("---")

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📈 Vue générale", "📊 Métriques", "🎯 Portefeuille", "📥 Export"])

with tab1:
    # Graphique des cours
    if not selected_prices.empty:
        st.subheader("📈 Évolution des cours")
        fig1 = px.line(selected_prices, title=f"Cours - {selected_year}")
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    # Rendements cumulés
    if not cum_returns.empty:
        st.subheader("📊 Rendements cumulés")
        fig2 = px.line(cum_returns, title="Performance cumulée")
        fig2.update_yaxes(tickformat=".0%")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # === SOLUTION POUR LA CARTE RENDEMENT/RISQUE ===
    st.subheader("🗺️ Carte rendement / risque")
    
    # Méthode alternative : utiliser un tableau + bar chart si scatter échoue
    if not metrics_df.empty and len(metrics_df) > 0:
        try:
            # Essayer d'abord avec scatter
            plot_ok = True
            for col in ["Risque", "Rendement", "Sharpe"]:
                if metrics_df[col].isna().all() or np.isinf(metrics_df[col]).any():
                    plot_ok = False
                    break
            
            if plot_ok and len(metrics_df) > 1:
                fig3 = px.scatter(
                    metrics_df,
                    x="Risque",
                    y="Rendement",
                    size="Sharpe",
                    color="Sharpe",
                    text="Banque",
                    title="Rendement vs Risque"
                )
                fig3.update_xaxes(tickformat=".0%")
                fig3.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                # Fallback: afficher un tableau et un bar chart
                st.info("Affichage alternatif de la carte rendement/risque")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.dataframe(metrics_df[["Banque", "Rendement", "Risque", "Sharpe"]].style.format({
                        "Rendement": "{:.2%}",
                        "Risque": "{:.2%}",
                        "Sharpe": "{:.3f}"
                    }), use_container_width=True)
                
                with col_b:
                    # Bar chart rendement par banque
                    fig_alt = px.bar(metrics_df, x="Banque", y="Rendement", title="Rendement par banque", color="Rendement")
                    fig_alt.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig_alt, use_container_width=True)
        
        except Exception as e:
            st.warning(f"Affichage simplifié: {str(e)[:100]}")
            st.dataframe(metrics_df[["Banque", "Rendement", "Risque"]].style.format({
                "Rendement": "{:.2%}",
                "Risque": "{:.2%}"
            }), use_container_width=True)

with tab2:
    st.subheader("📊 Métriques par banque")
    st.dataframe(
        metrics_df.style.format({
            "Rendement": "{:.2%}",
            "Risque": "{:.2%}",
            "Sharpe": "{:.3f}"
        }),
        use_container_width=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        fig_ret = px.bar(metrics_df, x="Banque", y="Rendement", title="Rendement annualisé", color="Rendement")
        fig_ret.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ret, use_container_width=True)
    
    with col2:
        fig_risk = px.bar(metrics_df, x="Banque", y="Risque", title="Volatilité annualisée", color="Risque")
        fig_risk.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_risk, use_container_width=True)

with tab3:
    st.subheader("🎯 Portefeuille optimal Sharpe maximum")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Rendement", f"{ret_opt:.2%}")
    col_b.metric("Risque", f"{vol_opt:.2%}")
    col_c.metric("Sharpe", f"{sharpe_opt:.3f}")
    
    st.markdown("---")
    
    col_pie, col_table = st.columns([1, 1.5])
    
    with col_pie:
        if len(weights_df) > 0:
            fig_pie = px.pie(weights_df.head(8), names="Banque", values="Poids", title="Répartition")
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_table:
        st.dataframe(
            weights_df.style.format({"Poids": "{:.2%}", "Montant (TND)": "{:,.2f}"}),
            use_container_width=True
        )
    
    # Graphique d'allocation
    if len(weights_df) > 0:
        fig_bar = px.bar(weights_df.head(8), x="Banque", y="Poids", title="Allocation", color="Poids")
        fig_bar.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Drawdown
    if not drawdown.empty:
        st.subheader("📉 Drawdown")
        fig_dd = px.area(drawdown, title="Perte maximale (%))")
        fig_dd.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_dd, use_container_width=True)

with tab4:
    st.subheader("📥 Export des résultats")
    
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            selected_prices.to_excel(writer, sheet_name="Prix")
            returns.to_excel(writer, sheet_name="Rendements")
            cum_returns.to_excel(writer, sheet_name="Rendements_cumules")
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

# ==============================
# FOOTER
# ==============================

st.markdown("---")
st.caption("⚠️ **Disclaimer :** Analyse à but éducatif. Ne constitue pas un conseil en investissement.")
st.caption(f"📊 Données {selected_year} | Méthodologie: Markowitz | Annualisation: 252 jours")
