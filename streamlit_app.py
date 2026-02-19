"""
📊 Dashboard de Análisis Fiscal con Enfoque de Redes
====================================================
Versión 4: Soporte para múltiples países y años

Ejecutar con: streamlit run fiscal_dashboard_v4.py
"""

import streamlit as st
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Análisis Fiscal - Redes I-O",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# IMPORTAR PLOTLY
# ============================================================================

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly no está instalado. Instálalo con: pip install plotly")

# ============================================================================
# TEXTOS DE DOCUMENTACIÓN
# ============================================================================

DOCS = {
    "intro": """
    ## 📊 Análisis del Impacto Fiscal con Enfoque de Redes
    
    Este dashboard implementa metodologías de **Input-Output** para analizar cómo los impuestos 
    y subsidios afectan la producción sectorial, considerando las **interdependencias entre sectores**.
    
    ### 🎯 Objetivo
    Medir el impacto sistémico de la política fiscal considerando que los sectores económicos 
    están conectados a través de cadenas de suministro.
    """,
    
    "tax_convention": """
    ### ⚠️ Convención del Campo `taxes_subsidies`
    
    | Signo | Interpretación | Ejemplo |
    |-------|---------------|---------|
    | **T > 0** | 🟢 **SUBSIDIO NETO** | Agricultura, Transporte público |
    | **T < 0** | 🔴 **IMPUESTO NETO** | Minería, Manufactura |
    """,
    
    "matrix_a": """
    ### 📐 Matriz A - Coeficientes Técnicos
    
    **Fórmula:** `A = Z × diag(X)⁻¹`
    
    **Interpretación:** `A[i,j]` = Cantidad de insumo del sector `i` necesario para producir **1 unidad** del sector `j`
    """,
    
    "matrix_l": """
    ### 🔄 Matriz L - Leontief
    
    **Fórmula:** `L = (I - A)⁻¹`
    
    **Interpretación:** `L[i,j]` = Producción **total** del sector `i` necesaria para satisfacer **1 unidad** de demanda final del sector `j`
    """,
    
    "multipliers": """
    ### 📊 Multiplicadores y Linkages
    
    - **Forward Linkage (FL):** Importancia como **proveedor**
    - **Backward Linkage (BL):** Importancia como **demandante**
    - **Multiplicador Tipo I:** Producción total generada por 1 unidad de demanda final
    """,
    
    "hef_method": """
    ### 🔬 Método HEF
    
    Simula qué pasaría si **eliminamos** el componente fiscal de cada sector.
    
    - Eliminar **IMPUESTO** (T<0) → Costos ↓ → Producción **↑**
    - Eliminar **SUBSIDIO** (T>0) → Costos ↑ → Producción **↓**
    """,
    
    "network_effects": """
    ### 🌐 Efecto de Red
    
    **Fórmula:** `NetEffect[j] = Σᵢ A[i,j] × t[i]`
    
    Impacto ponderado de los impuestos/subsidios de los **proveedores** del sector.
    """
}

# ============================================================================
# FUNCIONES DE DATOS MOCKUP
# ============================================================================

def generate_mockup_data():
    """Genera datos de ejemplo con múltiples países y años."""
    
    SECTORS = [
        'Agricultura', 'Minería', 'Manufactura', 'Energía',
        'Construcción', 'Comercio', 'Transporte', 'Servicios'
    ]
    
    COUNTRIES = ['ARG', 'BRA', 'CHL', 'COL', 'MEX', 'PER']
    YEARS = [2018, 2019, 2020]
    
    N = len(SECTORS)
    
    # Matriz Z base
    Z_base = np.array([
        [20, 5, 80, 2, 3, 10, 5, 5],
        [2, 30, 60, 15, 40, 2, 5, 3],
        [15, 10, 100, 20, 80, 30, 25, 40],
        [8, 25, 40, 10, 15, 20, 30, 25],
        [3, 5, 10, 5, 20, 8, 10, 15],
        [5, 3, 15, 3, 10, 15, 12, 20],
        [10, 15, 30, 8, 25, 18, 10, 15],
        [7, 12, 25, 12, 20, 25, 18, 30],
    ], dtype=float)
    
    Y_base = np.array([150, 80, 400, 120, 200, 250, 100, 300], dtype=float)
    T_base = np.array([+15.0, -25.0, -45.0, -10.0, -30.0, -20.0, +8.0, -35.0])
    
    connections_list = []
    variables_list = []
    
    np.random.seed(42)
    
    for country in COUNTRIES:
        # Factor de escala por país
        country_scale = np.random.uniform(0.5, 2.0)
        
        for year in YEARS:
            # Factor de crecimiento por año
            year_factor = 1 + (year - 2018) * 0.03
            
            # Ajustar matrices
            Z = Z_base * country_scale * year_factor * np.random.uniform(0.9, 1.1, (N, N))
            Y = Y_base * country_scale * year_factor * np.random.uniform(0.9, 1.1, N)
            T = T_base * country_scale * np.random.uniform(0.8, 1.2, N)
            
            X = Z.sum(axis=1) + Y
            VA = X - Z.sum(axis=0)
            VA_sin_T = VA - T
            Compensation = VA_sin_T * 0.6
            Operating_Surplus = VA_sin_T * 0.4
            
            # Crear conexiones
            for i in range(N):
                for j in range(N):
                    if Z[i, j] > 0:
                        connections_list.append({
                            'country_code': country,
                            'year': year,
                            'from_sector': SECTORS[i],
                            'to_sector': SECTORS[j],
                            'flow_value': Z[i, j]
                        })
            
            # Crear variables
            for i in range(N):
                variables_list.append({
                    'country_code': country,
                    'year': year,
                    'sector': SECTORS[i],
                    'gross_output': X[i],
                    'value_added': VA[i],
                    'taxes_subsidies': T[i],
                    'compensation': Compensation[i],
                    'operating_surplus': Operating_Surplus[i],
                    'final_demand': Y[i],
                    'exports': Y[i] * 0.3,
                    'imports': X[i] * 0.2
                })
    
    return pd.DataFrame(connections_list), pd.DataFrame(variables_list), SECTORS, COUNTRIES, YEARS

# ============================================================================
# CLASE DE ANÁLISIS (SOPORTA MÚLTIPLES PAÍSES/AÑOS)
# ============================================================================

class FiscalNetworkAnalyzer:
    """Analizador de redes I-O con soporte para múltiples países y años."""
    
    def __init__(self, connections_df, variables_df, countries=None, years=None):
        """
        Parameters:
        -----------
        connections_df : DataFrame con conexiones
        variables_df : DataFrame con variables sectoriales
        countries : list o str - País(es) a analizar
        years : list o int - Año(s) a analizar
        """
        self.connections_full = connections_df.copy()
        self.variables_full = variables_df.copy()
        
        # Convertir a listas si es necesario
        if countries is None:
            countries = connections_df['country_code'].unique().tolist()
        elif isinstance(countries, str):
            countries = [countries]
        
        if years is None:
            years = connections_df['year'].unique().tolist()
        elif isinstance(years, (int, float)):
            years = [int(years)]
        
        self.countries = countries
        self.years = years
        
        # Filtrar datos
        self.connections = connections_df[
            (connections_df['country_code'].isin(countries)) & 
            (connections_df['year'].isin(years))
        ].copy()
        
        self.variables = variables_df[
            (variables_df['country_code'].isin(countries)) & 
            (variables_df['year'].isin(years))
        ].copy()
        
        # Sectores únicos
        self.sectors = sorted(self.variables['sector'].unique())
        self.n_sectors = len(self.sectors)
        self.sector_to_idx = {s: i for i, s in enumerate(self.sectors)}
        
        # Construir matrices agregadas o por país-año
        self._build_matrices()
    
    def _build_matrices(self):
        """Construye matrices I-O (agregadas si hay múltiples países/años)."""
        n = self.n_sectors
        
        # Agregar flujos y variables
        # Si hay múltiples países/años, se suman los flujos
        
        # Matriz Z agregada
        self.Z = np.zeros((n, n))
        connections_grouped = self.connections.groupby(['from_sector', 'to_sector'])['flow_value'].sum()
        
        for (from_s, to_s), value in connections_grouped.items():
            i = self.sector_to_idx.get(from_s)
            j = self.sector_to_idx.get(to_s)
            if i is not None and j is not None:
                self.Z[i, j] = value
        
        # Vectores agregados
        variables_grouped = self.variables.groupby('sector').agg({
            'gross_output': 'sum',
            'final_demand': 'sum',
            'taxes_subsidies': 'sum',
            'value_added': 'sum'
        })
        
        self.X = np.zeros(n)
        self.Y = np.zeros(n)
        self.T = np.zeros(n)
        self.VA = np.zeros(n)
        
        for sector, row in variables_grouped.iterrows():
            idx = self.sector_to_idx.get(sector)
            if idx is not None:
                self.X[idx] = row['gross_output']
                self.Y[idx] = row['final_demand']
                self.T[idx] = row['taxes_subsidies']
                self.VA[idx] = row['value_added']
        
        # Matriz A
        X_inv = np.where(self.X > 0, 1/self.X, 0)
        self.A = self.Z @ np.diag(X_inv)
        
        # Matriz L
        try:
            self.L = np.linalg.inv(np.eye(n) - self.A)
        except Exception:
            self.L = np.linalg.pinv(np.eye(n) - self.A)
        
        # Tasa de impuesto
        self.tax_rate = np.where(self.X > 0, self.T / self.X, 0)
    
    def compute_multipliers(self):
        """Calcula multiplicadores y linkages."""
        forward = self.L.sum(axis=1)
        backward = self.L.sum(axis=0)
        
        fl_norm = forward / forward.mean() if forward.mean() > 0 else forward
        bl_norm = backward / backward.mean() if backward.mean() > 0 else backward
        
        classifications = []
        for i in range(self.n_sectors):
            if fl_norm[i] > 1 and bl_norm[i] > 1:
                classifications.append('Sector Clave')
            elif fl_norm[i] > 1:
                classifications.append('Forward Oriented')
            elif bl_norm[i] > 1:
                classifications.append('Backward Oriented')
            else:
                classifications.append('Linkages Débiles')
        
        return pd.DataFrame({
            'sector': self.sectors,
            'forward_linkage': forward,
            'backward_linkage': backward,
            'fl_normalized': fl_norm,
            'bl_normalized': bl_norm,
            'type_I_multiplier': backward,
            'classification': classifications
        })
    
    def fiscal_hypothetical_extraction(self, elasticity=0.5):
        """Ejecuta análisis HEF para todos los sectores."""
        results = []
        X_total_original = self.X.sum()
        
        for j in range(self.n_sectors):
            cost_change = -self.tax_rate[j]
            
            A_new = self.A.copy()
            for i in range(self.n_sectors):
                if self.A[j, i] > 0:
                    demand_change = elasticity * cost_change
                    A_new[j, i] = self.A[j, i] * (1 + demand_change)
            
            try:
                L_new = np.linalg.inv(np.eye(self.n_sectors) - A_new)
            except Exception:
                L_new = np.linalg.pinv(np.eye(self.n_sectors) - A_new)
            
            X_new = L_new @ self.Y
            X_total_new = X_new.sum()
            
            absolute_impact = X_total_new - X_total_original
            relative_impact = (absolute_impact / X_total_original) * 100 if X_total_original > 0 else 0
            
            results.append({
                'sector': self.sectors[j],
                'tax_original': self.T[j],
                'tax_type': 'Subsidio' if self.T[j] > 0 else 'Impuesto',
                'tax_rate_pct': self.tax_rate[j] * 100,
                'absolute_impact': absolute_impact,
                'relative_impact_pct': relative_impact,
                'X_new': X_new.tolist()
            })
        
        return pd.DataFrame(results)
    
    def simulate_shock(self, sector_idx, magnitude, n_iterations=10):
        """Simula propagación de shock."""
        epsilon = np.zeros(self.n_sectors)
        epsilon[sector_idx] = magnitude * self.X[sector_idx]
        
        W = self.A.T
        trajectory = [epsilon.sum()]
        cumulative = epsilon.copy()
        
        for _ in range(n_iterations):
            epsilon = W @ epsilon
            trajectory.append(epsilon.sum())
            cumulative += epsilon
        
        return {
            'trajectory': trajectory,
            'cumulative': cumulative,
            'total_effect': cumulative.sum()
        }
    
    def compute_network_effects(self):
        """Calcula efectos de red."""
        net_effect = self.A.T @ self.tax_rate
        
        return pd.DataFrame({
            'sector': self.sectors,
            'own_tax_rate': self.tax_rate * 100,
            'network_effect': net_effect * 100,
            'total_effect': (self.tax_rate + net_effect) * 100
        })
    
    def verify_calculations(self):
        """Verifica consistencia de cálculos."""
        checks = {
            'X = Z.sum(fila) + Y': np.allclose(self.X, self.Z.sum(axis=1) + self.Y),
            'X = L × Y': np.allclose(self.X, self.L @ self.Y),
            'Σ A[i,j] < 1 ∀j': (self.A.sum(axis=0) < 1).all() if self.A.sum() > 0 else True,
            'L ≥ 0': (self.L >= 0).all(),
            'L diagonal ≥ 1': (np.diag(self.L) >= 0.99).all()
        }
        return checks
    
    def get_comparison_by_country(self):
        """Obtiene métricas comparativas por país."""
        if len(self.countries) <= 1:
            return None
        
        results = []
        for country in self.countries:
            country_vars = self.variables[self.variables['country_code'] == country]
            
            if len(country_vars) == 0:
                continue
            
            total_output = country_vars['gross_output'].sum()
            total_tax = country_vars['taxes_subsidies'].sum()
            avg_tax_rate = (total_tax / total_output * 100) if total_output > 0 else 0
            
            results.append({
                'country': country,
                'total_output': total_output,
                'total_taxes': total_tax,
                'avg_tax_rate_pct': avg_tax_rate,
                'n_subsidized': (country_vars['taxes_subsidies'] > 0).sum(),
                'n_taxed': (country_vars['taxes_subsidies'] < 0).sum()
            })
        
        return pd.DataFrame(results)
    
    def get_comparison_by_year(self):
        """Obtiene métricas comparativas por año."""
        if len(self.years) <= 1:
            return None
        
        results = []
        for year in self.years:
            year_vars = self.variables[self.variables['year'] == year]
            
            if len(year_vars) == 0:
                continue
            
            total_output = year_vars['gross_output'].sum()
            total_tax = year_vars['taxes_subsidies'].sum()
            
            results.append({
                'year': year,
                'total_output': total_output,
                'total_taxes': total_tax,
                'avg_tax_rate_pct': (total_tax / total_output * 100) if total_output > 0 else 0
            })
        
        return pd.DataFrame(results)

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    st.title("📊 Análisis de Impacto Fiscal con Redes I-O")
    st.markdown("**Soporte para múltiples países y años**")
    
    # =========================================================================
    # SIDEBAR - CONFIGURACIÓN
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Fuente de datos
        data_source = st.radio(
            "Fuente de datos:",
            ["Datos de ejemplo", "Cargar mis datos CSV"]
        )
        
        if data_source == "Cargar mis datos CSV":
            st.markdown("---")
            connections_file = st.file_uploader("connections.csv", type=['csv'])
            variables_file = st.file_uploader("sector_variables.csv", type=['csv'])
            
            if connections_file and variables_file:
                try:
                    connections_df = pd.read_csv(connections_file)
                    variables_df = pd.read_csv(variables_file)
                    all_countries = sorted(connections_df['country_code'].unique().tolist())
                    all_years = sorted(connections_df['year'].unique().tolist())
                except Exception as e:
                    st.error(f"Error: {e}")
                    connections_df, variables_df, _, all_countries, all_years = generate_mockup_data()
            else:
                st.info("⬆️ Carga ambos archivos CSV")
                connections_df, variables_df, _, all_countries, all_years = generate_mockup_data()
        else:
            connections_df, variables_df, _, all_countries, all_years = generate_mockup_data()
        
        st.markdown("---")
        st.subheader("🌍 Selección de Datos")
        
        # Modo de selección
        selection_mode = st.radio(
            "Modo de análisis:",
            ["Un país, un año", "Múltiples países", "Múltiples años", "Múltiples países y años"],
            help="Selecciona cómo quieres agregar los datos"
        )
        
        # Selección según el modo
        if selection_mode == "Un país, un año":
            selected_countries = [st.selectbox("País:", all_countries)]
            selected_years = [st.selectbox("Año:", all_years)]
            
        elif selection_mode == "Múltiples países":
            selected_countries = st.multiselect(
                "Países:", 
                all_countries, 
                default=all_countries[:3] if len(all_countries) >= 3 else all_countries
            )
            selected_years = [st.selectbox("Año:", all_years)]
            
        elif selection_mode == "Múltiples años":
            selected_countries = [st.selectbox("País:", all_countries)]
            selected_years = st.multiselect(
                "Años:", 
                all_years, 
                default=all_years
            )
            
        else:  # Múltiples países y años
            selected_countries = st.multiselect(
                "Países:", 
                all_countries, 
                default=all_countries[:3] if len(all_countries) >= 3 else all_countries
            )
            selected_years = st.multiselect(
                "Años:", 
                all_years, 
                default=all_years
            )
        
        # Validar selección
        if not selected_countries:
            selected_countries = [all_countries[0]]
        if not selected_years:
            selected_years = [all_years[0]]
        
        st.markdown("---")
        st.subheader("🔧 Parámetros")
        elasticity = st.slider("Elasticidad HEF:", 0.1, 1.0, 0.5, 0.1)
        shock_magnitude = st.slider("Magnitud shock (%):", 1, 20, 10) / 100
        shock_iterations = st.slider("Rondas propagación:", 5, 20, 10)
        
        # Mostrar selección actual
        st.markdown("---")
        st.subheader("📋 Selección Actual")
        st.write(f"**Países:** {', '.join(selected_countries)}")
        st.write(f"**Años:** {', '.join(map(str, selected_years))}")
    
    # =========================================================================
    # CREAR ANALIZADOR
    # =========================================================================
    try:
        analyzer = FiscalNetworkAnalyzer(
            connections_df, 
            variables_df, 
            countries=selected_countries, 
            years=selected_years
        )
    except Exception as e:
        st.error(f"Error al crear el analizador: {e}")
        st.stop()
    
    # =========================================================================
    # TABS PRINCIPALES
    # =========================================================================
    tabs = st.tabs([
        "📋 Resumen", "🌍 Comparación", "🔢 Matrices", 
        "📊 Multiplicadores", "💰 HEF", "🌊 Propagación", "✅ Verificación"
    ])
    
    # =========================================================================
    # TAB 1: RESUMEN
    # =========================================================================
    with tabs[0]:
        st.header("📋 Resumen de Datos")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Países", len(selected_countries))
        with col2:
            st.metric("Años", len(selected_years))
        with col3:
            st.metric("Sectores", analyzer.n_sectors)
        with col4:
            st.metric("Producción Total", f"{analyzer.X.sum():,.0f} M")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Registros", f"{len(analyzer.variables):,}")
        with col2:
            st.metric("Conexiones", f"{len(analyzer.connections):,}")
        with col3:
            subsidios = analyzer.T[analyzer.T > 0].sum()
            st.metric("Total Subsidios", f"+{subsidios:,.0f} M")
        with col4:
            impuestos = abs(analyzer.T[analyzer.T < 0].sum())
            st.metric("Total Impuestos", f"-{impuestos:,.0f} M")
        
        st.markdown("---")
        
        # Datos agregados por sector
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Variables Agregadas por Sector")
            sector_summary = analyzer.variables.groupby('sector').agg({
                'gross_output': 'sum',
                'value_added': 'sum',
                'taxes_subsidies': 'sum',
                'final_demand': 'sum'
            }).round(2)
            st.dataframe(sector_summary, use_container_width=True)
        
        with col2:
            st.subheader("Distribución Fiscal")
            if PLOTLY_AVAILABLE:
                sector_tax = analyzer.variables.groupby('sector')['taxes_subsidies'].sum().reset_index()
                sector_tax['tipo'] = sector_tax['taxes_subsidies'].apply(lambda x: 'Subsidio' if x > 0 else 'Impuesto')
                
                fig = px.bar(
                    sector_tax.sort_values('taxes_subsidies'),
                    x='taxes_subsidies',
                    y='sector',
                    orientation='h',
                    color='tipo',
                    color_discrete_map={'Subsidio': '#28a745', 'Impuesto': '#dc3545'},
                    title='Componente Fiscal Agregado por Sector'
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Convención de signos"):
            st.markdown(DOCS["tax_convention"])
    
    # =========================================================================
    # TAB 2: COMPARACIÓN
    # =========================================================================
    with tabs[1]:
        st.header("🌍 Comparación entre Países/Años")
        
        # Comparación por país
        country_comparison = analyzer.get_comparison_by_country()
        if country_comparison is not None and len(country_comparison) > 0:
            st.subheader("📊 Comparación por País")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(country_comparison.round(2), use_container_width=True)
            
            with col2:
                if PLOTLY_AVAILABLE:
                    fig = px.bar(
                        country_comparison,
                        x='country',
                        y='total_output',
                        color='avg_tax_rate_pct',
                        color_continuous_scale='RdYlGn',
                        title='Producción Total por País'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            if PLOTLY_AVAILABLE:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        country_comparison,
                        x='country',
                        y=['n_subsidized', 'n_taxed'],
                        barmode='group',
                        title='Sectores Subsidiados vs Gravados por País',
                        labels={'value': 'Número de sectores', 'variable': 'Tipo'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        country_comparison,
                        x='country',
                        y='total_taxes',
                        color=country_comparison['total_taxes'].apply(lambda x: 'Subsidio Neto' if x > 0 else 'Impuesto Neto'),
                        title='Balance Fiscal por País'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecciona múltiples países para ver la comparación.")
        
        st.markdown("---")
        
        # Comparación por año
        year_comparison = analyzer.get_comparison_by_year()
        if year_comparison is not None and len(year_comparison) > 0:
            st.subheader("📈 Evolución Temporal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(year_comparison.round(2), use_container_width=True)
            
            with col2:
                if PLOTLY_AVAILABLE:
                    fig = px.line(
                        year_comparison,
                        x='year',
                        y='total_output',
                        markers=True,
                        title='Evolución de la Producción Total'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            if PLOTLY_AVAILABLE:
                fig = px.line(
                    year_comparison,
                    x='year',
                    y=['total_output', 'total_taxes'],
                    markers=True,
                    title='Evolución de Producción y Balance Fiscal'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecciona múltiples años para ver la evolución temporal.")
    
    # =========================================================================
    # TAB 3: MATRICES
    # =========================================================================
    with tabs[2]:
        st.header("🔢 Matrices Input-Output")
        
        if len(selected_countries) > 1 or len(selected_years) > 1:
            st.info(f"⚠️ Matrices agregadas para {len(selected_countries)} país(es) y {len(selected_years)} año(s)")
        
        matrix_choice = st.selectbox(
            "Seleccionar matriz:",
            ["Z - Consumo Intermedio", "A - Coeficientes Técnicos", "L - Leontief"]
        )
        
        if matrix_choice == "Z - Consumo Intermedio":
            matrix_data = analyzer.Z
            with st.expander("ℹ️ ¿Qué es la Matriz Z?"):
                st.markdown("**Z[i,j]** = Cuánto compra el sector j del sector i")
        elif matrix_choice == "A - Coeficientes Técnicos":
            matrix_data = analyzer.A
            with st.expander("ℹ️ ¿Qué es la Matriz A?"):
                st.markdown(DOCS["matrix_a"])
        else:
            matrix_data = analyzer.L
            with st.expander("ℹ️ ¿Qué es la Matriz L?"):
                st.markdown(DOCS["matrix_l"])
        
        if PLOTLY_AVAILABLE:
            fig = px.imshow(
                matrix_data,
                x=analyzer.sectors,
                y=analyzer.sectors,
                color_continuous_scale='Blues',
                title=matrix_choice
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Ver tabla"):
            df_matrix = pd.DataFrame(matrix_data, index=analyzer.sectors, columns=analyzer.sectors)
            st.dataframe(df_matrix.round(4), use_container_width=True)
    
    # =========================================================================
    # TAB 4: MULTIPLICADORES
    # =========================================================================
    with tabs[3]:
        st.header("📊 Multiplicadores y Clasificación Sectorial")
        
        with st.expander("ℹ️ Metodología"):
            st.markdown(DOCS["multipliers"])
        
        multipliers_df = analyzer.compute_multipliers()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(multipliers_df.round(4), use_container_width=True)
        
        with col2:
            max_idx = multipliers_df['type_I_multiplier'].idxmax()
            min_idx = multipliers_df['type_I_multiplier'].idxmin()
            
            st.metric("Mayor Multiplicador", 
                     f"{multipliers_df.loc[max_idx, 'sector']}",
                     f"{multipliers_df.loc[max_idx, 'type_I_multiplier']:.4f}")
            st.metric("Menor Multiplicador",
                     f"{multipliers_df.loc[min_idx, 'sector']}",
                     f"{multipliers_df.loc[min_idx, 'type_I_multiplier']:.4f}")
        
        if PLOTLY_AVAILABLE:
            fig = px.scatter(
                multipliers_df,
                x='bl_normalized',
                y='fl_normalized',
                text='sector',
                color='classification',
                size='type_I_multiplier',
                color_discrete_map={
                    'Sector Clave': '#dc3545',
                    'Forward Oriented': '#007bff',
                    'Backward Oriented': '#28a745',
                    'Linkages Débiles': '#6c757d'
                },
                title='Clasificación Sectorial'
            )
            fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_traces(textposition='top center')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 5: HEF
    # =========================================================================
    with tabs[4]:
        st.header("💰 Extracción Hipotética Fiscal (HEF)")
        
        with st.expander("ℹ️ Metodología"):
            st.markdown(DOCS["hef_method"])
        
        hef_results = analyzer.fiscal_hypothetical_extraction(elasticity=elasticity)
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_hef = hef_results[['sector', 'tax_type', 'tax_original', 'tax_rate_pct', 'relative_impact_pct']]
            st.dataframe(display_hef.round(4), use_container_width=True)
        
        with col2:
            max_idx = hef_results['relative_impact_pct'].abs().idxmax()
            max_row = hef_results.loc[max_idx]
            
            st.metric("Mayor Impacto Sistémico",
                     max_row['sector'],
                     f"{max_row['relative_impact_pct']:+.4f}%")
        
        if PLOTLY_AVAILABLE:
            col1, col2 = st.columns(2)
            
            with col1:
                df_sorted = hef_results.sort_values('relative_impact_pct')
                fig = px.bar(
                    df_sorted,
                    x='relative_impact_pct',
                    y='sector',
                    orientation='h',
                    color='tax_type',
                    color_discrete_map={'Subsidio': '#28a745', 'Impuesto': '#dc3545'},
                    title='Impacto de Eliminar Componente Fiscal'
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    hef_results,
                    x='tax_original',
                    y='relative_impact_pct',
                    text='sector',
                    color='tax_type',
                    color_discrete_map={'Subsidio': '#28a745', 'Impuesto': '#dc3545'},
                    title='T Original vs Impacto HEF'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 6: PROPAGACIÓN
    # =========================================================================
    with tabs[5]:
        st.header("🌊 Propagación de Shocks")
        
        with st.expander("ℹ️ Metodología"):
            st.markdown(DOCS["network_effects"])
        
        shock_sector = st.selectbox("Sector origen:", analyzer.sectors, key='shock')
        shock_idx = analyzer.sector_to_idx[shock_sector]
        
        shock_result = analyzer.simulate_shock(shock_idx, shock_magnitude, shock_iterations)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Shock Inicial", f"{shock_magnitude * analyzer.X[shock_idx]:,.2f} M")
        with col2:
            st.metric("Efecto Total", f"{shock_result['total_effect']:,.2f} M")
        with col3:
            initial = shock_magnitude * analyzer.X[shock_idx]
            ratio = shock_result['total_effect'] / initial if initial > 0 else 0
            st.metric("Multiplicador", f"{ratio:.4f}x")
        
        if PLOTLY_AVAILABLE:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(shock_result['trajectory']))),
                    y=shock_result['trajectory'],
                    mode='lines+markers',
                    fill='tozeroy'
                ))
                fig.update_layout(title='Propagación por Ronda', xaxis_title='Ronda', yaxis_title='Efecto')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cumulative = shock_result['cumulative']
                sorted_idx = np.argsort(cumulative)[::-1]
                colors = ['orange' if analyzer.sectors[i] == shock_sector else 'steelblue' for i in sorted_idx]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=cumulative[sorted_idx],
                    y=[analyzer.sectors[i] for i in sorted_idx],
                    orientation='h',
                    marker_color=colors
                ))
                fig.update_layout(title='Efecto Acumulado por Sector')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🌐 Efectos de Red")
        
        network_effects = analyzer.compute_network_effects()
        st.dataframe(network_effects.round(2), use_container_width=True)
    
    # =========================================================================
    # TAB 7: VERIFICACIÓN
    # =========================================================================
    with tabs[6]:
        st.header("✅ Verificación")
        
        checks = analyzer.verify_calculations()
        
        for name, passed in checks.items():
            if passed:
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Balance Fiscal")
            st.write(f"**Subsidios (T>0):** {(analyzer.T > 0).sum()} sectores")
            st.write(f"**Impuestos (T<0):** {(analyzer.T < 0).sum()} sectores")
            st.write(f"**Balance neto:** {analyzer.T.sum():+,.2f} M")
        
        with col2:
            st.subheader("Exportar")
            
            mult_csv = analyzer.compute_multipliers().to_csv(index=False)
            st.download_button("📊 Multiplicadores", mult_csv, "multiplicadores.csv", "text/csv")
            
            hef_csv = analyzer.fiscal_hypothetical_extraction().drop(columns=['X_new']).to_csv(index=False)
            st.download_button("💰 Resultados HEF", hef_csv, "hef_results.csv", "text/csv")
            
            net_csv = analyzer.compute_network_effects().to_csv(index=False)
            st.download_button("🌐 Efectos de Red", net_csv, "network_effects.csv", "text/csv")

# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == "__main__":
    main()
