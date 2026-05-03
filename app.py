import io
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy import stats
import streamlit as st

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Markowitz BVMT - Suite Complète", layout="wide")

st.title("📊 Tableau de bord Markowitz BVMT")
st.markdown("### Optimisation de portefeuille - Analyse financière avancée")
st.markdown("---")

# ==============================
# BANQUES TUNISIENNES
# ==============================

BANQUES = [
    "BIAT", "ATB", "STB", "BT", "AMEN BANK", "UIB", "UBCI", "BH",
    "BNA", "ATTIJARI BANK", "BH BANK", "BTE", "WIFACK INT BANK"
]

# ==============================
# CHARGEMENT DES DONNÉES
# ==============================

@st.cache_data
def load_and_filter(file_path, year):
    """Charge et filtre les données"""
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        
        # Identifier les colonnes
        col_date = None
        col_nom = None
        col_prix = None
        
        for col in df.columns:
            if col.upper() in ['SEANCE', 'DATE']:
                col_date = col
            elif col.upper() in ['VALEUR', 'VALUEUR', 'SOCIETE']:
                col_nom = col
            elif col.upper() in ['CLOTURE', 'CLOSE', 'PRIX']:
                col_prix = col
        
        if not all([col_date, col_nom, col_prix]):
            return None
        
        df = df[[col_date, col_nom, col_prix]].copy()
        df.columns = ['Date', 'Societe', 'Prix']
        
        # Nettoyage
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['Date'])
        
        df['Prix'] = df['Prix'].astype(str).str.replace(',', '.')
        df['Prix'] = pd.to_numeric(df['Prix'], errors='coerce')
        df = df.dropna(subset=['Prix'])
        df = df[df['Prix'] > 0]
        
        # Filtre année
        df['Annee'] = df['Date'].dt.year
        df = df[df['Annee'] == year]
        
        # Filtre banques
        df['Societe'] = df['Societe'].astype(str).str.strip()
        df = df[df['Societe'].str.upper().isin([b.upper() for b in BANQUES])]
        
        return df
    
    except Exception as e:
        return None


@st.cache_data
def create_prices(data):
    """Crée la matrice des prix"""
    if data is None or data.empty:
        return None
    
    prices = data.pivot_table(
        index='Date',
        columns='Societe',
        values='Prix',
        aggfunc='first'
    ).sort_index().ffill()
    
    return prices


# ==============================
# FONCTIONS FINANCIÈRES
# ==============================

def calculate_var(returns, confidence=0.95):
    """Value at Risk"""
    return returns.quantile(1 - confidence) * np.sqrt(252)


def calculate_cvar(returns, confidence=0.95):
    """Conditional Value at Risk (Expected Shortfall)"""
    var = returns.quantile(1 - confidence)
    return returns[returns <= var].mean() * np.sqrt(252)


def calculate_beta(returns, market_returns):
    """Beta par rapport au marché"""
    if len(market_returns) > 1:
        cov = np.cov(returns, market_returns)[0, 1]
        var_market = np.var(market_returns)
        if var_market > 0:
            return cov / var_market
    return 1


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
    
    def min_variance(w):
        return port_vol(w)
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    # Portefeuille Sharpe max
    try:
        result_sharpe = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500})
        weights_sharpe = result_sharpe.x if result_sharpe.success else init
    except:
        weights_sharpe = init
    
    # Portefeuille variance minimale
    try:
        result_minvar = minimize(min_variance, init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500})
        weights_minvar = result_minvar.x if result_minvar.success else init
    except:
        weights_minvar = init
    
    # Normaliser
    weights_sharpe = np.maximum(weights_sharpe, 0)
    weights_sharpe = weights_sharpe / weights_sharpe.sum() if weights_sharpe.sum() > 0 else init
    
    weights_minvar = np.maximum(weights_minvar, 0)
    weights_minvar = weights_minvar / weights_minvar.sum() if weights_minvar.sum() > 0 else init
    
    ret_sharpe = port_return(weights_sharpe)
    vol_sharpe = port_vol(weights_sharpe)
    sharpe_ratio = (ret_sharpe - rf) / vol_sharpe if vol_sharpe > 0 else 0
    
    ret_minvar = port_return(weights_minvar)
    vol_minvar = port_vol(weights_minvar)
    sharpe_minvar = (ret_minvar - rf) / vol_minvar if vol_minvar > 0 else 0
    
    return weights_sharpe, ret_sharpe, vol_sharpe, sharpe_ratio, weights_minvar, ret_minvar, vol_minvar, sharpe_minvar


def efficient_frontier(mean_returns, cov_matrix, rf, n_points=20):
    """Calcul de la frontière efficiente"""
    frontier_returns = []
    frontier_risks = []
    
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    n = len(mean_returns)
    init = np.ones(n) / n
    
    def port_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    
    for target in target_returns:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.sum(w * mean_returns) - t}
        )
        try:
            result = minimize(port_vol, init, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500})
            if result.success:
                frontier_returns.append(target)
                frontier_risks.append(result.fun)
        except:
            pass
    
    return frontier_returns, frontier_risks


def calculate_portfolio_var(weights, returns, confidence=0.95):
    """VaR du portefeuille"""
    port_returns = returns.dot(weights)
    return port_returns.quantile(1 - confidence) * np.sqrt(252)


# ==============================
# MAIN
# ==============================

BASE_DIR = Path(__file__).parent
YEARS = [2023, 2024, 2025]

# Sidebar
st.sidebar.header("⚙️ Configuration")

selected_year = st.sidebar.selectbox("📅 Année à analyser", YEARS, index=len(YEARS)-1)

# Trouver le fichier
file_path = None
for f in BASE_DIR.iterdir():
    if f.is_file() and f.suffix in ['.xlsx', '.xls']:
        if str(selected_year) in f.stem:
            file_path = f
            break

if not file_path:
    st.error(f"Fichier {selected_year}.xlsx non trouvé")
    st.stop()

# Chargement
with st.spinner(f"📂 Chargement {selected_year}..."):
    data = load_and_filter(file_path, selected_year)
    if data is None or data.empty:
        st.error("Aucune donnée trouvée")
        st.stop()
    
    prices = create_prices(data)
    if prices is None or prices.empty:
        st.error("Erreur création prix")
        st.stop()

st.sidebar.success(f"✅ {len(prices.columns)} banques chargées")

# Sélection banques
selected_banks = st.sidebar.multiselect(
    "🏦 Banques à analyser",
    options=sorted(prices.columns),
    default=sorted(prices.columns)[:min(8, len(prices.columns))]
)

if len(selected_banks) < 2:
    selected_banks = sorted(prices.columns)[:2]

# Paramètres
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Paramètres")
rf = st.sidebar.number_input("Taux sans risque (%)", value=7.5, step=0.5) / 100
capital = st.sidebar.number_input("Capital (TND)", value=10000, step=5000)

# ==============================
# CALCULS PRINCIPAUX
# ==============================

with st.spinner("🔄 Calculs en cours..."):
    selected_prices = prices[selected_banks].dropna(how='all').ffill()
    returns = selected_prices.pct_change().dropna()
    
    if returns.empty or len(returns) < 10:
        st.error("Données insuffisantes")
        st.stop()
    
    # Métriques annualisées
    mean_returns = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()
    
    # Nettoyage
    mean_returns = mean_returns.fillna(0).replace([np.inf, -np.inf], 0)
    volatility = volatility.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Sharpe individuel
    sharpe_individual = (mean_returns - rf) / volatility
    sharpe_individual = sharpe_individual.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Rendements cumulés
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # Drawdown
    cummax = selected_prices.cummax()
    drawdown = (selected_prices - cummax) / cummax
    drawdown = drawdown.fillna(0)
    max_drawdown = drawdown.min()
    
    # VaR
    var_90 = calculate_var(returns, 0.90)
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    
    # CVaR / Expected Shortfall
    cvar_95 = calculate_cvar(returns, 0.95)
    
    # Beta marché
    market_returns = returns.mean(axis=1)
    betas = {}
    for bank in returns.columns:
        betas[bank] = calculate_beta(returns[bank], market_returns)
    betas = pd.Series(betas)
    
    # Optimisation
    (w_sharpe, ret_sharpe, vol_sharpe, sharpe_opt,
     w_minvar, ret_minvar, vol_minvar, sharpe_minvar) = optimize_portfolio(mean_returns, cov_matrix, rf)
    
    # Frontière efficiente
    frontier_returns, frontier_risks = efficient_frontier(mean_returns, cov_matrix, rf)
    
    # VaR des portefeuilles
    port_var_sharpe = calculate_portfolio_var(w_sharpe, returns, 0.95)
    port_var_minvar = calculate_portfolio_var(w_minvar, returns, 0.95)
    
    # DataFrames
    weights_sharpe_df = pd.DataFrame({
        'Banque': selected_banks,
        'Poids': w_sharpe,
        'Montant (TND)': w_sharpe * capital
    })
    weights_sharpe_df = weights_sharpe_df[weights_sharpe_df['Poids'] > 0.001].sort_values('Poids', ascending=False)
    
    weights_minvar_df = pd.DataFrame({
        'Banque': selected_banks,
        'Poids': w_minvar,
        'Montant (TND)': w_minvar * capital
    })
    weights_minvar_df = weights_minvar_df[weights_minvar_df['Poids'] > 0.001].sort_values('Poids', ascending=False)
    
    # Métriques individuelles
    metrics_df = pd.DataFrame({
        'Banque': selected_banks,
        'Rendement annualisé': mean_returns.values,
        'Volatilité': volatility.values,
        'Sharpe': sharpe_individual.values,
        'Drawdown max': max_drawdown.values * 100,
        'VaR 90%': var_90.values * 100,
        'VaR 95%': var_95.values * 100,
        'VaR 99%': var_99.values * 100,
        'CVaR 95%': cvar_95.values * 100,
        'Beta': betas.values
    })
    metrics_df = metrics_df.round(4)
    
    # Classement intelligent
    metrics_df['Score'] = (
        metrics_df['Sharpe'].rank(ascending=False) +
        metrics_df['Rendement annualisé'].rank(ascending=False) +
        metrics_df['Volatilité'].rank(ascending=True) +
        metrics_df['Drawdown max'].rank(ascending=True) +
        metrics_df['VaR 95%'].rank(ascending=True)
    )
    metrics_df = metrics_df.sort_values('Score').reset_index(drop=True)
    best_bank = metrics_df.iloc[0]['Banque'] if len(metrics_df) > 0 else 'N/A'
    
    # Recommandations
    def get_recommendation(row):
        if row['Sharpe'] > 1 and row['Volatilité'] < metrics_df['Volatilité'].median():
            return '🟢 Très attractive'
        elif row['Sharpe'] > 0.5:
            return '🟡 Intéressante'
        elif row['Volatilité'] > metrics_df['Volatilité'].median():
            return '🔴 Risque élevé'
        else:
            return '⚪ À surveiller'
    
    metrics_df['Recommandation'] = metrics_df.apply(get_recommendation, axis=1)

# ==============================
# INTERFACE - 10 ONGLETS
# ==============================

t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
    "📈 Vue générale",
    "📊 Métriques",
    "📉 Risques & VaR",
    "📈 CML & SML",
    "🎯 Optimisation",
    "📉 Frontière efficiente",
    "🤖 Recommandations",
    "💼 Simulation",
    "🔗 Corrélations",
    "📥 Export"
])

with t1:
    st.header(f"📈 Vue générale - {selected_year}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Banques", len(selected_banks))
    c2.metric("Meilleure banque", best_bank)
    c3.metric("Sharpe optimal", f"{sharpe_opt:.3f}")
    c4.metric("Capital", f"{capital:,.0f} TND")
    
    # Évolution des cours
    st.subheader("📈 Évolution des cours")
    fig_prices = px.line(selected_prices, title=f"Cours de clôture - {selected_year}")
    fig_prices.update_layout(height=450)
    st.plotly_chart(fig_prices, use_container_width=True)
    
    # Rendements cumulés
    st.subheader("📊 Rendements cumulés")
    fig_cum = px.line(cumulative_returns, title="Performance cumulée")
    fig_cum.update_yaxes(tickformat='.0%')
    fig_cum.update_layout(height=400)
    st.plotly_chart(fig_cum, use_container_width=True)
    
    # Carte rendement/risque - VERSION SANS SIZE POUR ÉVITER L'ERREUR
    st.subheader("🗺️ Carte rendement / risque")
    plot_df = metrics_df.copy()
    plot_df['Rendement annualisé'] = plot_df['Rendement annualisé'].clip(-0.5, 0.5)
    plot_df['Volatilité'] = plot_df['Volatilité'].clip(0, 1)
    
    # Scatter SANS le paramètre size pour éviter l'erreur
    fig_scatter = px.scatter(
        plot_df,
        x='Volatilité',
        y='Rendement annualisé',
        color='Sharpe',
        text='Banque',
        title='Positionnement des banques',
        labels={'Volatilité': 'Risque (Volatilité)', 'Rendement annualisé': 'Rendement annualisé'},
        color_continuous_scale='RdYlGn'
    )
    fig_scatter.update_traces(textposition='top center', marker=dict(size=18))
    fig_scatter.update_xaxes(tickformat='.0%')
    fig_scatter.update_yaxes(tickformat='.0%')
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

with t2:
    st.header("📊 Métriques détaillées")
    
    st.dataframe(
        metrics_df.style.format({
            'Rendement annualisé': '{:.2%}',
            'Volatilité': '{:.2%}',
            'Sharpe': '{:.3f}',
            'Drawdown max': '{:.1f}%',
            'VaR 90%': '{:.1f}%',
            'VaR 95%': '{:.1f}%',
            'VaR 99%': '{:.1f}%',
            'CVaR 95%': '{:.1f}%',
            'Beta': '{:.2f}',
            'Score': '{:.0f}'
        }),
        use_container_width=True,
        height=400
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ret = px.bar(metrics_df, x='Banque', y='Rendement annualisé', title='Rendement annualisé', color='Rendement annualisé', color_continuous_scale='Greens')
        fig_ret.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_ret, use_container_width=True)
    
    with col2:
        fig_sharpe = px.bar(metrics_df, x='Banque', y='Sharpe', title='Ratio de Sharpe', color='Sharpe', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_sharpe, use_container_width=True)

with t3:
    st.header("⚠️ Analyse des risques")
    
    # Drawdown
    st.subheader("📉 Drawdown des banques")
    fig_dd = px.area(drawdown, title="Drawdown", labels={"value": "Perte (%)", "variable": "Banque"})
    fig_dd.update_yaxes(tickformat='.0%')
    fig_dd.update_layout(height=400)
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Comparaison VaR
    st.subheader("📊 Comparaison Value at Risk")
    var_df = pd.DataFrame({
        'VaR 90%': var_90.values * 100,
        'VaR 95%': var_95.values * 100,
        'VaR 99%': var_99.values * 100
    }, index=selected_banks)
    
    fig_var = px.bar(var_df, title="VaR par niveau de confiance", barmode='group')
    fig_var.update_yaxes(title='VaR (%)')
    st.plotly_chart(fig_var, use_container_width=True)
    
    # Expected Shortfall
    st.subheader("📊 Expected Shortfall (CVaR) à 95%")
    fig_cvar = px.bar(metrics_df, x='Banque', y='CVaR 95%', title='Expected Shortfall 95%', color='CVaR 95%', color_continuous_scale='Reds')
    fig_cvar.update_yaxes(title='CVaR (%)')
    st.plotly_chart(fig_cvar, use_container_width=True)
    
    # Distribution avec VaR pour une banque sélectionnée
    st.subheader("📈 Distribution des rendements avec VaR")
    selected_bank_var = st.selectbox("Choisir une banque", selected_banks, key='var_select')
    
    if selected_bank_var:
        returns_bank = returns[selected_bank_var].dropna()
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=returns_bank, name='Rendements', nbinsx=50, opacity=0.7))
        
        # Lignes VaR
        for conf, color, label in [(0.90, 'orange', 'VaR 90%'), (0.95, 'red', 'VaR 95%'), (0.99, 'darkred', 'VaR 99%')]:
            var_val = returns_bank.quantile(1 - conf)
            fig_dist.add_vline(x=var_val, line_dash='dash', line_color=color, annotation_text=f'{label}: {var_val:.2%}')
        
        # Densité
        try:
            kde = stats.gaussian_kde(returns_bank)
            x_range = np.linspace(returns_bank.min(), returns_bank.max(), 100)
            fig_dist.add_trace(go.Scatter(x=x_range, y=kde(x_range) * len(returns_bank) * (returns_bank.max() - returns_bank.min()) / 50, name='Densité', line=dict(color='blue')))
        except:
            pass
        
        fig_dist.update_layout(title=f'Distribution - {selected_bank_var}', xaxis_title='Rendement', yaxis_title='Fréquence', height=500)
        fig_dist.update_xaxes(tickformat='.1%')
        st.plotly_chart(fig_dist, use_container_width=True)

with t4:
    st.header("📈 Capital Market Line (CML) & Security Market Line (SML)")
    
    # CML
    st.subheader("📊 Capital Market Line (CML)")
    
    if vol_sharpe > 0:
        cml_risks = np.linspace(0, max(volatility.max(), vol_sharpe) * 1.5, 50)
        cml_returns = rf + (ret_sharpe - rf) / vol_sharpe * cml_risks
        
        fig_cml = go.Figure()
        fig_cml.add_trace(go.Scatter(x=cml_risks, y=cml_returns, mode='lines', name='CML', line=dict(color='green', width=3, dash='dash')))
        fig_cml.add_trace(go.Scatter(x=[vol_sharpe], y=[ret_sharpe], mode='markers', name='Portefeuille marché', marker=dict(size=15, color='red', symbol='star')))
        fig_cml.add_trace(go.Scatter(x=[0], y=[rf], mode='markers', name='Actif sans risque', marker=dict(size=12, color='blue')))
        fig_cml.add_trace(go.Scatter(x=volatility, y=mean_returns, mode='markers', name='Banques', marker=dict(size=8, color='gray'), text=selected_banks, hoverinfo='text'))
        
        fig_cml.update_layout(title='Capital Market Line', xaxis_title='Risque (Volatilité)', yaxis_title='Rendement attendu', height=500)
        fig_cml.update_xaxes(tickformat='.0%')
        fig_cml.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_cml, use_container_width=True)
    
    # SML
    st.subheader("📈 Security Market Line (SML)")
    
    if not betas.empty:
        sml_betas = np.linspace(0, max(betas.max() * 1.2, 1.5), 50)
        sml_returns = rf + (ret_sharpe - rf) * sml_betas
        
        fig_sml = go.Figure()
        fig_sml.add_trace(go.Scatter(x=sml_betas, y=sml_returns, mode='lines', name='SML', line=dict(color='purple', width=3, dash='dash')))
        fig_sml.add_trace(go.Scatter(x=[1], y=[ret_sharpe], mode='markers', name='Marché (β=1)', marker=dict(size=15, color='red', symbol='star')))
        fig_sml.add_trace(go.Scatter(x=[0], y=[rf], mode='markers', name='Sans risque (β=0)', marker=dict(size=12, color='blue')))
        fig_sml.add_trace(go.Scatter(x=betas, y=mean_returns, mode='markers', name='Banques', marker=dict(size=10, color='gray'), text=[f"{b}<br>β: {beta:.2f}" for b, beta in zip(selected_banks, betas)], hoverinfo='text'))
        
        fig_sml.update_layout(title='Security Market Line - CAPM', xaxis_title='Beta', yaxis_title='Rendement attendu', height=500)
        fig_sml.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_sml, use_container_width=True)
        
        st.dataframe(pd.DataFrame({'Banque': betas.index, 'Beta': betas.values, 'Interprétation': ['Défensif' if b < 1 else 'Agressif' for b in betas.values]}).style.format({'Beta': '{:.3f}'}), use_container_width=True)

with t5:
    st.header("🎯 Optimisation Markowitz")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Portefeuille Sharpe Maximum")
        st.metric("Rendement", f"{ret_sharpe:.2%}")
        st.metric("Risque", f"{vol_sharpe:.2%}")
        st.metric("Sharpe", f"{sharpe_opt:.4f}")
        st.metric("VaR 95%", f"{port_var_sharpe:.2%}")
        
        st.subheader("Allocation")
        if not weights_sharpe_df.empty:
            fig_pie1 = px.pie(weights_sharpe_df.head(8), names='Banque', values='Poids', title='Répartition Sharpe max')
            st.plotly_chart(fig_pie1, use_container_width=True)
    
    with col2:
        st.subheader("🛡️ Portefeuille Variance Minimale")
        st.metric("Rendement", f"{ret_minvar:.2%}")
        st.metric("Risque", f"{vol_minvar:.2%}")
        st.metric("Sharpe", f"{sharpe_minvar:.4f}")
        st.metric("VaR 95%", f"{port_var_minvar:.2%}")
        
        st.subheader("Allocation")
        if not weights_minvar_df.empty:
            fig_pie2 = px.pie(weights_minvar_df.head(8), names='Banque', values='Poids', title='Répartition Variance min')
            st.plotly_chart(fig_pie2, use_container_width=True)
    
    # Tableau comparatif
    st.subheader("📊 Comparaison des portefeuilles")
    comparison_df = pd.DataFrame({
        'Métrique': ['Rendement', 'Risque', 'Sharpe', 'VaR 95%'],
        'Sharpe Max': [f'{ret_sharpe:.2%}', f'{vol_sharpe:.2%}', f'{sharpe_opt:.4f}', f'{port_var_sharpe:.2%}'],
        'Variance Min': [f'{ret_minvar:.2%}', f'{vol_minvar:.2%}', f'{sharpe_minvar:.4f}', f'{port_var_minvar:.2%}']
    })
    st.dataframe(comparison_df, use_container_width=True)

with t6:
    st.header("📉 Frontière efficiente de Markowitz")
    
    if len(frontier_returns) > 0:
        fig_frontier = go.Figure()
        
        # Frontière
        fig_frontier.add_trace(go.Scatter(
            x=frontier_risks, y=frontier_returns,
            mode='lines+markers', name='Frontière efficiente',
            line=dict(color='blue', width=2), marker=dict(size=5)
        ))
        
        # Portefeuille Sharpe max
        fig_frontier.add_trace(go.Scatter(
            x=[vol_sharpe], y=[ret_sharpe],
            mode='markers', name='Sharpe max',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        # Portefeuille variance min
        fig_frontier.add_trace(go.Scatter(
            x=[vol_minvar], y=[ret_minvar],
            mode='markers', name='Variance min',
            marker=dict(size=15, color='green', symbol='triangle-up')
        ))
        
        # Banques individuelles
        fig_frontier.add_trace(go.Scatter(
            x=volatility, y=mean_returns,
            mode='markers', name='Banques',
            marker=dict(size=10, color='gray', symbol='circle'),
            text=selected_banks, hoverinfo='text'
        ))
        
        fig_frontier.update_layout(
            title='Frontière efficiente',
            xaxis_title='Risque (Volatilité annualisée)',
            yaxis_title='Rendement annualisé',
            height=600
        )
        fig_frontier.update_xaxes(tickformat='.0%')
        fig_frontier.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_frontier, use_container_width=True)
        
        st.info("""
        **Interprétation :**
        - La courbe bleue représente les portefeuilles optimaux
        - ⭐ Point rouge : Portefeuille qui maximise le ratio de Sharpe
        - ▲ Point vert : Portefeuille de variance minimale
        - ● Points gris : Banques individuelles
        """)
    else:
        st.info("Frontière efficiente non disponible")

with t7:
    st.header("🤖 Recommandations intelligentes")
    
    st.info("Analyse basée sur les données historiques - Non constitutif d'un conseil financier")
    st.success(f"🏆 **Meilleure banque selon le modèle : {best_bank}**")
    
    st.subheader("Classement des banques")
    st.dataframe(
        metrics_df[['Banque', 'Rendement annualisé', 'Volatilité', 'Sharpe', 'Drawdown max', 'VaR 95%', 'Beta', 'Score', 'Recommandation']].style.format({
            'Rendement annualisé': '{:.2%}',
            'Volatilité': '{:.2%}',
            'Sharpe': '{:.3f}',
            'Drawdown max': '{:.1f}%',
            'VaR 95%': '{:.1f}%',
            'Beta': '{:.2f}',
            'Score': '{:.0f}'
        }),
        use_container_width=True
    )
    
    # Graphique score
    fig_score = px.bar(metrics_df, x='Banque', y='Score', color='Recommandation', title='Score de qualité')
    st.plotly_chart(fig_score, use_container_width=True)
    
    st.warning("""
    ⚠️ **Avertissement :** 
    - Analyse quantitative uniquement
    - Vérifiez liquidité, frais, fiscalité avant investissement
    - Consultez un conseiller professionnel
    """)

with t8:
    st.header("💼 Simulation d'investissement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portefeuille Sharpe Max")
        st.dataframe(weights_sharpe_df.style.format({'Poids': '{:.2%}', 'Montant (TND)': '{:,.2f}'}), use_container_width=True)
        
        if not weights_sharpe_df.empty:
            fig_inv1 = px.bar(weights_sharpe_df.head(8), x='Banque', y='Montant (TND)', title='Montants à investir - Sharpe max', color='Montant (TND)')
            st.plotly_chart(fig_inv1, use_container_width=True)
    
    with col2:
        st.subheader("Portefeuille Variance Min")
        st.dataframe(weights_minvar_df.style.format({'Poids': '{:.2%}', 'Montant (TND)': '{:,.2f}'}), use_container_width=True)
        
        if not weights_minvar_df.empty:
            fig_inv2 = px.bar(weights_minvar_df.head(8), x='Banque', y='Montant (TND)', title='Montants à investir - Variance min', color='Montant (TND)')
            st.plotly_chart(fig_inv2, use_container_width=True)
    
    # Profil investisseur
    st.subheader("Recommandation personnalisée")
    profile = st.selectbox("Votre profil", ["Prudent", "Équilibré", "Dynamique"])
    
    if profile == "Prudent":
        st.success("✅ Portefeuille à variance minimale recommandé")
        st.metric("Rentabilité attendue", f"{ret_minvar:.2%}")
        st.metric("Risque attendu", f"{vol_minvar:.2%}")
        st.metric("VaR 95%", f"{port_var_minvar:.2%}")
    elif profile == "Équilibré":
        mix_ret = (ret_sharpe + ret_minvar) / 2
        mix_vol = (vol_sharpe + vol_minvar) / 2
        st.success("✅ Mixte (50% Sharpe max + 50% Variance min) recommandé")
        st.metric("Rentabilité attendue", f"{mix_ret:.2%}")
        st.metric("Risque attendu", f"{mix_vol:.2%}")
    else:
        st.success("✅ Portefeuille Sharpe maximum recommandé")
        st.metric("Rentabilité attendue", f"{ret_sharpe:.2%}")
        st.metric("Risque attendu", f"{vol_sharpe:.2%}")
        st.metric("VaR 95%", f"{port_var_sharpe:.2%}")

with t9:
    st.header("🔗 Matrices de corrélation et covariance")
    
    st.subheader("Matrice de corrélation")
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect='auto',
        title='Corrélations entre banques',
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.subheader("Matrice de covariance annualisée")
    fig_cov = px.imshow(
        cov_matrix,
        text_auto=True,
        aspect='auto',
        title='Covariance annualisée',
        color_continuous_scale='Viridis'
    )
    fig_cov.update_layout(height=600)
    st.plotly_chart(fig_cov, use_container_width=True)

with t10:
    st.header("📥 Export des résultats")
    
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            selected_prices.to_excel(writer, sheet_name='1_Prix')
            returns.to_excel(writer, sheet_name='2_Rendements')
            cumulative_returns.to_excel(writer, sheet_name='3_Rendements_cumules')
            metrics_df.to_excel(writer, sheet_name='4_Metriques_banques', index=False)
            weights_sharpe_df.to_excel(writer, sheet_name='5_Allocation_Sharpe_max', index=False)
            weights_minvar_df.to_excel(writer, sheet_name='6_Allocation_Variance_min', index=False)
            corr_matrix.to_excel(writer, sheet_name='7_Matrice_correlation')
            cov_matrix.to_excel(writer, sheet_name='8_Matrice_covariance')
            drawdown.to_excel(writer, sheet_name='9_Drawdown')
            
            # Statistiques portfolio
            portfolio_stats = pd.DataFrame({
                'Métrique': ['Rendement Sharpe max', 'Risque Sharpe max', 'Sharpe ratio', 'VaR 95% Sharpe max',
                            'Rendement Variance min', 'Risque Variance min', 'Sharpe ratio Variance min', 'VaR 95% Variance min'],
                'Valeur': [f'{ret_sharpe:.2%}', f'{vol_sharpe:.2%}', f'{sharpe_opt:.4f}', f'{port_var_sharpe:.2%}',
                          f'{ret_minvar:.2%}', f'{vol_minvar:.2%}', f'{sharpe_minvar:.4f}', f'{port_var_minvar:.2%}']
            })
            portfolio_stats.to_excel(writer, sheet_name='10_Stats_portefeuille', index=False)
        
        st.download_button(
            label="📥 Télécharger le rapport Excel complet",
            data=output.getvalue(),
            file_name=f"markowitz_bvmt_{selected_year}.xlsx",
            use_container_width=True
        )
        
        st.success("✅ Le rapport contient toutes les analyses:")
        st.markdown("""
        - Prix historiques
        - Rendements journaliers et cumulés
        - Métriques individuelles (Sharpe, VaR 90/95/99%, CVaR, Beta, Drawdown)
        - Allocations optimales (Sharpe max et Variance min)
        - Matrices de corrélation et covariance
        - Drawdown
        - Statistiques des portefeuilles
        """)
    
    except Exception as e:
        st.error(f"Erreur export: {e}")

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer :** Analyse à but éducatif - Consultez un professionnel avant d'investir")
st.caption(f"📊 Méthodologie Markowitz | Annualisation 252 jours | Données BVMT {selected_year}")
