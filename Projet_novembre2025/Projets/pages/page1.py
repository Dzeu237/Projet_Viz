import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Cosmétiques",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title(" Dashboard Analyse Produits Cosmétiques")
st.markdown("---")

st.markdown("""
    Suite à la réalisation d'un bilan d'entreprise exhaustif, notre mission est d'analyser la situation de l'entreprise X. Il est essentiel de présenter de manière synthétique les forces et faiblesses identifiées, en s'appuyant sur des visuels pertinents et didactiques. Ces analyses et visualisations doivent être immédiatement exploitables pour faciliter la prise de décision stratégique par les dirigeants.

Afin d'assurer une lecture structurée et une action ciblée, nous segmenterons l'ensemble des indicateurs et visuels selon les quatre axes stratégiques suivants :
 """)

st.markdown("---")

st.session_state.current_page= st.session_state.get('current_page', 'Finance')
# Navigation horizontale
col_nav2, col_nav3, col_nav4, col_nav5 = st.columns([ 1, 1, 1, 1])

with col_nav2:
    if st.button("💶 Financial Metrics", key="Financial_metrics", width='content'):
        st.session_state.current_page = 'Finance'


with col_nav3:
    if st.button("📋 Operational Metrics", key="Operational_metrics", width='content'):
        st.session_state.current_page = 'Operational'


with col_nav4:
    if st.button("☑️ Quality Metrics", key="Quality_metrics", width='content'):
        st.session_state.current_page = 'Quality'


with col_nav5:
    if st.button("🤝 Commercial Metrics", key="Commercial_metrics", width='content'):
        st.session_state.current_page = 'Commercial'


st.markdown("---")

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Chargement des données
@st.cache_data
def load_data():
    path="https://github.com/Dzeu237/Projet_Viz/blob/main/Projet_novembre2025/Projets/Data/supply_chain_data.csv?raw=true"
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()  # Nettoyage des noms de colonnes
    data.rename(columns={'Product type':'Product_type',
                         'Inspection results':'Inspection_results',
                         'Customer demographics':'Customer_demographics',
                         'Revenue generated':'Revenue_generated',
                         'Number of products sold':'Number_of_products_sold',
                         'Defect rates':'Defect_rates',
                         'Stock levels':'Stock_levels',
                         'Manufacturing costs':'Manufacturing_costs_by_unit',
                         'Production volumes':'Production_volumes',
                         'Manufacturing lead time':'Manufacturing_lead_time',
                         'Order quantities':'Order_quantities',
                         'Shipping costs':'Shipping_costs_by_unit',
                         'Shipping times':'Shipping_times',
                         'Lead time':'Approvisionnement_lead_time',
                        }, inplace=True)  # Renommer les colonnes si nécessaire
    return pd.DataFrame(data)

df = load_data()



#Transform columns
   # Total Sold
df['Chiffre_d_affaires'] = df['Price'] * df['Number_of_products_sold'] # df['Revenue_generated']
df.drop(columns=['Revenue_generated','Lead times'], inplace=True)
    # Variable Cost by units
df['Manufacturing_costs'] = df['Manufacturing_costs_by_unit'] * df['Production_volumes']
df['Shipping_costs'] = df['Shipping_costs_by_unit'] * df['Number_of_products_sold']
df['Variable_Cost'] = df['Manufacturing_costs'] + df['Shipping_costs']
df['Lead_Time_Total'] = df['Manufacturing_lead_time'] + df['Approvisionnement_lead_time'] # Lead Time Approvissionement et Production
    # Fixed Cost by units
df.rename(columns={'Costs':'Fixed_cost'}, inplace=True)

#Creation des filtres
product_types = df['Product_type'].unique().tolist()
demographics = df['Customer_demographics'].unique().tolist()
localisations = df['Location'].unique().tolist()
df_filtered = df.copy()
col1,col2,col3=st.columns(3)
selected_product_types = col1.selectbox("Sélectionner le type de produit", options=["Tous"] + product_types)
if selected_product_types != "Tous":
    df_filtered = df_filtered[df_filtered['Product_type'] == selected_product_types]
selected_demographics = col2.selectbox("Sélectionner la démographie client", options=["Tous"] + demographics)
if selected_demographics != "Tous":
    df_filtered = df_filtered[df_filtered['Customer_demographics'] == selected_demographics]
selected_localisations = col3.selectbox("Sélectionner la localisation", options=["Tous"] + localisations)
if selected_localisations != "Tous":
    df_filtered = df_filtered[df_filtered['Location'] == selected_localisations]



#Formules pour les metriques
    #Formules Metriques Financieres
#Marge Contributive (€) = Chiffre d'affaires - (Manufacturing_costs × Number_of_products_sold)
chiffre_affaires = df_filtered['Chiffre_d_affaires'].sum()
Marge_Commerciale = chiffre_affaires - df_filtered['Variable_Cost'].sum()
#Marge Contributive (%) = (Marge Commerciale (€) / Chiffre d'affaires) × 100
Taux_de_Marge_Commerciale = (Marge_Commerciale / chiffre_affaires)*100
#Resultat = Marge Commerciale - Fixed_cost 
Resultat = Marge_Commerciale - df_filtered['Fixed_cost'].sum()
# ROI = (Revenue_generated - Manufacturing_costs × Production_volumes) / (Manufacturing_costs × Production_volumes) × 100
Total_Cost = df_filtered['Variable_Cost'].sum() + df_filtered['Fixed_cost'].sum()
ROI = ((chiffre_affaires - Total_Cost) / Total_Cost)*100
#CCC = Jour_appro + Jour_production + DIO + Jour_transport + DSO - DPO (CASH CONVERSION CYCLE)
Jour_appro = df_filtered['Approvisionnement_lead_time'].mean()
Jour_production = df_filtered['Manufacturing_lead_time'].mean()
DIO = (df_filtered['Stock_levels'].mean() / df_filtered['Number_of_products_sold'].mean()) * 365
Jour_transport = df_filtered['Shipping_times'].mean()
DSO = 30 # Supposons 30 jours pour DSO
DPO = 45 # Supposons 45 jours pour DPO
CCC=Jour_appro + Jour_production + DIO + Jour_transport + DSO - DPO

    #Formules Metriques Operationnelles
#Taux d'utilisation = (Number_of_products_sold / Production_volumes) × 100
Taux_d_utilisation = (df_filtered['Number_of_products_sold'].sum() / df_filtered['Production_volumes'].sum()) * 100
#Revenue_Efficiency = Revenue_generated / (Manufacturing_costs_by_unit × Number_of_products_sold)
Average_Revenue_Efficiency = (df_filtered['Chiffre_d_affaires'].mean() / df_filtered['Variable_Cost'].mean())*100
#Sales_Velocity = Number_of_products_sold / (Approvisionnement_lead_time + Manufacturing_lead_time)
Sales_Velocity = df_filtered['Number_of_products_sold'].mean() / (Jour_production + Jour_transport)
#Production_Velocity = Production_volumes / (Manufacturing_lead_time + Approvisionnement_lead_time)
Production_Velocity = df_filtered['Production_volumes'].mean() / (Jour_production + Jour_appro)
#Indice de Complexité Opérationnelle
Operational_Complexity_Index = ((Jour_transport / 7) + (Jour_appro / 7) + (Jour_production / 7)).mean()
#Pourcentage de Disponibilite
Avabality=df_filtered['Availability'].mean()

    #Formules Metriques Qualite
#Taux de Défaut
Taux_Défaut = df_filtered['Defect_rates'].mean()
#Taux de Réussite d'Inspection
Taux_Réussite_Inspection = (df_filtered[df_filtered['Inspection_results'] == 'Pass'].shape[0] / df_filtered.shape[0]) * 100

st.subheader("Metrique de performance")

if st.session_state.current_page == 'Finance':
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Chiffre d'affiares ", value=f'{chiffre_affaires:,.0f} €')
    col1.metric(label='CCC (Cash Conversion Cycle)', value=f'{CCC:,.0f} jours')
    col2.metric(label="Taux de Marge Commerciale", value=f'{Taux_de_Marge_Commerciale:,.2f} %')
    col3.metric(label='Resultat', value=f'{Resultat:,.2f} €')
    col3.metric(label='ROI', value=f'{ROI:,.2f} %')
    
elif st.session_state.current_page == 'Operational':
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Taux d'utilisation", value=f'{Taux_d_utilisation:,.2f} %')
    col2.metric(label='Productivité Revenu Moyen', value=f'{Average_Revenue_Efficiency:,.2f} %')
    col3.metric(label='Indice de Complexité Opérationnelle', value=f'{Operational_Complexity_Index:,.2f}')
    col1.metric(label="Sales Velocity", value=f'{Sales_Velocity:,.0f} unités/jour')
    col2.metric(label='Disponibilité Moyenne', value=f'{Avabality:,.2f} %')
    col3.metric(label='Production Velocity', value=f'{Production_Velocity:,.0f} unités/jour')

elif st.session_state.current_page == 'Quality':
    col1, col2, col3 = st.columns(3)
    defect_rate = df_filtered['Defect_rates'].mean()
    col1.metric(label="Taux de Défaut Moyen", value=f'{defect_rate:,.2f} %')
    col2.metric(label="Taux de Réussite d'Inspection", value=f'{Taux_Réussite_Inspection:,.2f} %')

st.markdown("---")
st.write(df.head(3))
st.title("Visualisations")
if st.session_state.current_page == 'Finance':
    st.subheader("Visualisations des Metrics Financières")
    col1, col2 = st.columns([0.45, 0.55])

    with col1:
        # Graphique 1: Chiffre d'affaires par Type de Produit
        revenue_by_product = df.groupby('Product_type')['Chiffre_d_affaires'].sum().reset_index().sort_values(by='Chiffre_d_affaires', ascending=False)
        cout_variable_by_product = df.groupby('Product_type')['Variable_Cost'].sum().reset_index().sort_values(by='Variable_Cost', ascending=False)
        fig1=go.Figure(data=[
            go.Bar(x=revenue_by_product['Product_type'], y=revenue_by_product['Chiffre_d_affaires'], name='Chiffre d\'affaires',text=[f'{val:,.0f} €' for val in revenue_by_product['Chiffre_d_affaires']],textposition='outside', marker_color='indianred'),
            go.Bar(x=cout_variable_by_product['Product_type'], y=cout_variable_by_product['Variable_Cost'], name='Coût Variable',text=[f'{val:,.0f} €' for val in cout_variable_by_product['Variable_Cost']], textposition='outside',marker_color='lightsalmon')
        ]
        )
        fig1.update_layout(
            title="Chiffre d'affaires et Coût Variable par Type de Produit",
            xaxis_title="Type de Produit",
            yaxis_title="Montant (€)",
            barmode='group'
        )
        st.plotly_chart(fig1, width='content')

        #Graphique 2: Circle chart des produits vendus
        product_sales = df.groupby('Product_type')['Number_of_products_sold'].sum().reset_index()
        fig2 = px.pie(product_sales, values='Number_of_products_sold', names='Product_type',
                    title='Répartition des Produits Vendus par Type de Produit',
                    hole=0.4)
        st.plotly_chart(fig2, width='content')

    with col2:
        # Graphique 3: graphique en cascade Taux de Marge par produit
        marge_by_product = df.groupby('Product_type').apply(lambda x: ((x['Chiffre_d_affaires'].sum() - x['Variable_Cost'].sum())/x['Chiffre_d_affaires'].sum())*100).reset_index(name='Marge_Commerciale').sort_values(by='Marge_Commerciale', ascending=False)
        fig3 = go.Figure(go.Waterfall(
            name="Taux de Marge par Type de Produit",
            orientation="v",
            measure=["relative"] * len(marge_by_product),
            x=marge_by_product['Product_type'],
            y=marge_by_product['Marge_Commerciale'],
            text=[f'{val:,.0f} %' for val in marge_by_product['Marge_Commerciale']],
            textposition="outside",
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig3.update_layout(
            title="Taux de Marge Commerciale par Type de Produit",
            xaxis_title="Type de Produit",
            yaxis_title="Taux de Marge Commerciale (%)",
        )
        st.plotly_chart(fig3, width='content')

        # Graphique 4: ROI par type de produit
        roi_by_product = df.groupby('Product_type').apply(lambda x: ((x['Chiffre_d_affaires'].sum() - (x['Variable_Cost'].sum() + x['Fixed_cost'].sum())) / (x['Variable_Cost'].sum() + x['Fixed_cost'].sum())) * 100).reset_index(name='ROI').sort_values(by='ROI', ascending=False)
        fig4 = go.Figure(go.Funnel(
            x=roi_by_product['Product_type'],
            y=roi_by_product['ROI'],
            text=[f'{val:,.2f} %' for val in roi_by_product['ROI']],
            textposition="outside",
            marker={"color": "teal"}
        ))
        fig4.update_layout(
            title="ROI par Type de Produit",
            xaxis_title="ROI (%)",
            yaxis_title="Type de Produit",
        )
        st.plotly_chart(fig4, width='content')

# Visualisations des Metrics Operationnelles
elif st.session_state.current_page == 'Operational':
    st.subheader("Visualisations des Metrics Opérationnelles")
    col1,col2=st.columns(2)

    with col1:
# Graphique 1: bar graphe des couts de transport moyen par route, type de transport,localisation
        transport_mode=df[['Location','Routes','Transportation modes','Shipping_costs_by_unit']].groupby(['Location','Routes','Transportation modes']).apply(lambda x:pd.Series({
                'Shippment_mean_Cost':x['Shipping_costs_by_unit'].mean()
            })).reset_index()
        fig1=px.bar(transport_mode,y='Location',x='Shippment_mean_Cost',color='Routes',barmode='group',orientation='h')
        fig1.update_traces(text=transport_mode['Transportation modes'],textposition='inside')
        st.plotly_chart(fig1,width='content')
# Graphique 2: Sales Velocity et Production Velocity
        velocity_by_product = df.groupby('Product_type').apply(lambda x: pd.Series({
            'Sales_Velocity': x['Number_of_products_sold'].mean() / (x['Manufacturing_lead_time'].mean() + x['Shipping_times'].mean()),
            'Production_Velocity': x['Production_volumes'].mean() / (x['Manufacturing_lead_time'].mean() + x['Approvisionnement_lead_time'].mean())
        })).reset_index()
        fig2=go.Figure(data=[
            go.Bar(name='Sales Velocity', x=velocity_by_product['Product_type'], y=velocity_by_product['Sales_Velocity'], text=[f'{val:,.0f} unités/jour' for val in velocity_by_product['Sales_Velocity']], textposition='outside', marker_color='slateblue'),
            go.Bar(name='Production Velocity', x=velocity_by_product['Product_type'], y=velocity_by_product['Production_Velocity'], text=[f'{val:,.0f} unités/jour' for val in velocity_by_product['Production_Velocity']], textposition='outside', marker_color='orchid')
        ])
        fig2.update_layout(
            title="Sales Velocity et Production Velocity par Type de Produit",
            xaxis_title="Type de Produit",
            barmode='group'
        )
        st.plotly_chart(fig2, width='content')

    with col2:
        # Graphique 3: Indice de Complexité Opérationnelle par type de produit
        complexity_by_product = df.groupby('Product_type').apply(lambda x: ((x['Shipping_times'].mean() / 7) + (x['Approvisionnement_lead_time'].mean() / 7) + (x['Manufacturing_lead_time'].mean() / 7))).reset_index(name='Operational_Complexity_Index').sort_values(by='Operational_Complexity_Index', ascending=False)
        fig3 = px.bar(complexity_by_product, x='Product_type', y='Operational_Complexity_Index',
                      title='Indice de Complexité Opérationnelle par Type de Produit',
                      text=[f'{val:,.2f}' for val in complexity_by_product['Operational_Complexity_Index']]
                      )
        fig3.update_traces(textposition='outside', marker_color='coral')
        st.plotly_chart(fig3, width='content')
        # Graphique 4: Disponibilité et Stock Level Moyenne par type de produit
        availability_by_product = df.groupby('Product_type').apply(lambda x: pd.Series({
            'Availability': x['Availability'].mean(),
            'Stock_levels': x['Stock_levels'].mean()
        })).reset_index()
        fig4=go.Figure(data=[
            go.Bar(name='Disponibilité', x=availability_by_product['Product_type'], y=availability_by_product['Availability'], text=[f'{val:,.2f} %' for val in availability_by_product['Availability']], textposition='outside', marker_color='lightseagreen'),
            go.Bar(name='Stock Level', x=availability_by_product['Product_type'], y=availability_by_product['Stock_levels'], text=[f'{val:,.0f} %' for val in availability_by_product['Stock_levels']], textposition='outside', marker_color='goldenrod')
        ])
        fig4.update_layout(
            title="Disponibilité et Stock Level Moyen par Type de Produit",
            xaxis_title="Type de Produit",
            barmode='group'
        )
        st.plotly_chart(fig4, width='content')
elif st.session_state.current_page == 'Quality':
    st.subheader("Visualisations des Metrics Qualité")
    col1, col2 = st.columns(2)
    with col1:
        # Graphique 1: Taux de Défaut par type de produit
        defect_rate_by_product = df.groupby('Product_type')['Defect_rates'].mean().reset_index().sort_values(by='Defect_rates', ascending=False)
        fig1 = px.bar(defect_rate_by_product, x='Product_type', y='Defect_rates',
                      title='Taux de Défaut par Type de Produit',
                      text=[f'{val:,.2f} %' for val in defect_rate_by_product['Defect_rates']]
                      )
        fig1.update_traces(textposition='outside', marker_color='salmon')
        st.plotly_chart(fig1, width='content')
    with col2:
        # Graphique 2: Taux de Réussite d'Inspection par type de produit
        inspection_success_by_product = df.groupby('Product_type').apply(lambda x: (x[x['Inspection_results'] == 'Pass'].shape[0] / x.shape[0]) * 100).reset_index(name='Taux_Réussite_Inspection').sort_values(by='Taux_Réussite_Inspection', ascending=False)
        fig2 = px.bar(inspection_success_by_product, x='Product_type', y='Taux_Réussite_Inspection',
                      title='Taux de Réussite d\'Inspection par Type de Produit',
                      text=[f'{val:,.2f} %' for val in inspection_success_by_product['Taux_Réussite_Inspection']]
                      )
        fig2.update_traces(textposition='outside', marker_color='mediumseagreen')
        st.plotly_chart(fig2, width='content')
if st.session_state.current_page == 'Commercial':
    st.subheader("Visualisations des Metrics Commerciales")
    

else:
    st.info("Veuillez sélectionner l'onglet 'Financial Metrics' pour afficher ces visualisations.")
    st.write(st.session_state.current_page)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Dashboard créé avec Streamlit 💄| Propulsé par Python 🐍</div>",
    unsafe_allow_html=True
)