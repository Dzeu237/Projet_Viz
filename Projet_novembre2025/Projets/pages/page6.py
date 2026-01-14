import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Analyse Risques Portefeuille", layout="wide")
st.title("📊 Analyse de Risques de Portefeuille Actions")

# Sidebar - Configuration
st.sidebar.header("Configuration")
tickers_input = st.sidebar.text_input("Tickers (séparés par virgule)", "AAPL,MSFT,GOOGL,AMZN,TSLA")
weights_input = st.sidebar.text_input("Poids (%) - optionnel", "")
period = st.sidebar.selectbox("Période", ["1mo", "2mo", "3mo", "5mo"], index=1)
risk_free_rate = st.sidebar.number_input("Taux sans risque (%)", value=4.0, step=0.1) / 100

tickers = [t.strip().upper() for t in tickers_input.split(",")]
if weights_input:
    weights = np.array([float(w) for w in weights_input.split(",")])
    weights = weights / weights.sum()
else:
    weights = np.array([1/len(tickers)] * len(tickers))

# Récupération des données
@st.cache_data(ttl=3600)
def get_data(tickers, period):
    data = yf.download(tickers, period=period, progress=False)['Close']
    return data

with st.spinner("Récupération des données..."):
    prices = get_data(tickers, period)
    returns = prices.pct_change().dropna()

# Calculs de risque
portfolio_returns = (returns * weights).sum(axis=1)
portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
portfolio_mean_return = portfolio_returns.mean() * 252
sharpe_ratio = (portfolio_mean_return - risk_free_rate) / portfolio_volatility
var_95 = np.percentile(portfolio_returns, 5)
cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
correlation_matrix = returns.corr()

# Métriques principales
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Rendement Annualisé", f"{portfolio_mean_return*100:.2f}%")
col2.metric("Volatilité Annuelle", f"{portfolio_volatility*100:.2f}%")
col3.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")
col4.metric("VaR 95% (1j)", f"{var_95*100:.2f}%", delta_color="inverse")
col5.metric("CVaR 95%", f"{cvar_95*100:.2f}%", delta_color="inverse")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🔗 Corrélations", "⚠️ Stress Tests", "📊 Distribution"])

with tab1:
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Evolution du portefeuille
        portfolio_value = (1 + portfolio_returns).cumprod()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value.values, 
                                  mode='lines', name='Portefeuille', line=dict(color='blue', width=2)))
        fig1.update_layout(title="Évolution de la Valeur du Portefeuille", 
                          xaxis_title="Date", yaxis_title="Valeur Normalisée (Base 1)")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_b:
        # Contribution par actif
        individual_vols = returns.std() * np.sqrt(252) * 100
        fig2 = go.Figure(data=[go.Bar(x=tickers, y=individual_vols.values)])
        fig2.update_layout(title="Volatilité Annualisée par Actif (%)", 
                          xaxis_title="Actif", yaxis_title="Volatilité (%)")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Poids du portefeuille
    fig3 = go.Figure(data=[go.Pie(labels=tickers, values=weights*100, hole=0.3)])
    fig3.update_layout(title="Allocation du Portefeuille")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    # Matrice de corrélation
    fig4 = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    fig4.update_layout(title="Matrice de Corrélation des Rendements", height=500)
    st.plotly_chart(fig4, use_container_width=True)
    
    # Alerte corrélations élevées
    high_corr = []
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                high_corr.append(f"{tickers[i]} - {tickers[j]}: {correlation_matrix.iloc[i, j]:.2f}")
    
    if high_corr:
        st.warning("⚠️ Corrélations élevées détectées (>0.7):")
        for corr in high_corr:
            st.write(f"• {corr}")

with tab3:
    st.subheader("Simulation de Stress Tests")
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        shock_market = st.slider("Choc de marché (%)", -30, 0, -10)
    with col_d:
        vol_multiplier = st.slider("Multiplicateur de volatilité", 1.0, 3.0, 2.0, 0.1)
    
    # Simulation choc marché
    shocked_portfolio = portfolio_value.iloc[-1] * (1 + shock_market/100)
    impact = (shocked_portfolio - portfolio_value.iloc[-1]) / portfolio_value.iloc[-1] * 100
    
    # Simulation volatilité
    shocked_vol = portfolio_volatility * vol_multiplier
    shocked_var = np.percentile(portfolio_returns * vol_multiplier, 5)
    
    col_e, col_f, col_g = st.columns(3)
    col_e.metric("Valeur Portefeuille Post-Choc", f"{shocked_portfolio:.2f}", f"{impact:.2f}%")
    col_f.metric("Volatilité Stressée", f"{shocked_vol*100:.2f}%", f"+{(vol_multiplier-1)*100:.0f}%")
    col_g.metric("VaR 95% Stressée", f"{shocked_var*100:.2f}%", delta_color="inverse")
    
    # Visualisation scénarios
    scenarios = pd.DataFrame({
        'Scénario': ['Normal', 'Choc Marché', 'Vol. Doublée', 'Combiné'],
        'VaR 95% (%)': [var_95*100, var_95*100*(1+shock_market/100), 
                        shocked_var*100, shocked_var*100*(1+shock_market/100)]
    })
    
    fig5 = px.bar(scenarios, x='Scénario', y='VaR 95% (%)', 
                  title="Comparaison des Scénarios de Stress", color='VaR 95% (%)',
                  color_continuous_scale='Reds')
    st.plotly_chart(fig5, use_container_width=True)

with tab4:
    col_h, col_i = st.columns(2)
    
    with col_h:
        # Distribution des rendements
        fig6 = go.Figure()
        fig6.add_trace(go.Histogram(x=portfolio_returns*100, nbinsx=50, 
                                    name='Rendements', marker_color='lightblue'))
        fig6.add_vline(x=var_95*100, line_dash="dash", line_color="red", 
                       annotation_text=f"VaR 95%: {var_95*100:.2f}%")
        fig6.update_layout(title="Distribution des Rendements Quotidiens (%)", 
                          xaxis_title="Rendement (%)", yaxis_title="Fréquence")
        st.plotly_chart(fig6, use_container_width=True)
    
    with col_i:
        # Rolling VaR
        rolling_var = portfolio_returns.rolling(window=60).quantile(0.05) * 100
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=rolling_var.index, y=rolling_var.values, 
                                  mode='lines', name='VaR 95% Rolling (60j)', 
                                  line=dict(color='red')))
        fig7.update_layout(title="Évolution de la VaR 95% (Fenêtre 60 jours)", 
                          xaxis_title="Date", yaxis_title="VaR (%)")
        st.plotly_chart(fig7, use_container_width=True)

# Alertes
st.subheader("🚨 Alertes")
alerts = []
if portfolio_volatility > 0.30:
    alerts.append("⚠️ Volatilité élevée (>30%)")
if sharpe_ratio < 0.5:
    alerts.append("⚠️ Ratio de Sharpe faible (<0.5)")
if abs(var_95) > 0.05:
    alerts.append("⚠️ VaR 95% dépassement seuil (>5%)")

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.success("✅ Aucune alerte - Portefeuille dans les seuils normaux")

# Données brutes
with st.expander("📋 Voir les données brutes"):
    st.dataframe(returns.tail(10).style.format("{:.2%}"))