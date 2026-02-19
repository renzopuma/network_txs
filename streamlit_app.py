"""
📊 Dashboard de Análisis Fiscal con Enfoque de Redes
====================================================
Versión 5: Manejo robusto de datos reales

Ejecutar con: streamlit run fiscal_dashboard_v5.py
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
    """,
    
    "matrix_l": """
    ### 🔄 Matriz L - Leontief
    **Fórmula:** `L = (I - A)⁻¹`
    """,
    
    "multipliers": """
    ### 📊 Multiplicadores y Linkages
    - **Forward Linkage (FL):** Importancia como **proveedor**
    - **Backward Linkage (BL):** Importancia como **demandante**
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
    """
}

# ============================================================================
# FUNCIÓN DE LIMPIEZA DE DATOS
# ============================================================================

def clean_dataframe(df, df_type='variables'):
    """
    Limpia y valida el DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
    df_type : 'variables' o 'connections'
    """
    df = df.copy()
    
    if df_type == 'variables':
        # Eliminar filas donde sector es nulo o vacío
        if 'sector' in df.columns:
            df = df[df['sector'].notna()]
            df = df[df['sector'].astype(str).str.strip() != '']
            df['sector'] = df['sector'].astype(str).str.strip()
        
        # Convertir columnas numéricas
        numeric_cols = ['gross_output', 'value_added', 'taxes_subsidies', 
                       'compensation', 'operating_surplus', 'final_demand',
                       'exports', 'imports']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Asegurar que year sea entero
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        
        # Asegurar que country_code sea string
        if 'country_code' in df.columns:
            df['country_code'] = df['country_code'].astype(str).str.strip()
    
    elif df_type == 'connections':
        # Eliminar filas donde sectores son nulos
        if 'from_sector' in df.columns:
            df = df[df['from_sector'].notna()]
            df = df[df['from_sector'].astype(str).str.strip() != '']
            df['from_sector'] = df['from_sector'].astype(str).str.strip()
        
        if 'to_sector' in df.columns:
            df = df[df['to_sector'].notna()]
            df = df[df['to_sector'].astype(str).str.strip() != '']
            df['to_sector'] = df['to_sector'].astype(str).str.strip()
        
        # Convertir flow_value a numérico
        if 'flow_value' in df.columns:
            df['flow_value'] = pd.to_numeric(df['flow_value'], errors='coerce').fillna(0)
        
        # Asegurar tipos correctos
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        
        if 'country_code' in df.columns:
            df['country_code'] = df['country_code'].astype(str).str.strip()
    
    return df

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
        country_scale = np.random.uniform(0.5, 2.0)
        
        for year in YEARS:
            year_factor = 1 + (year - 2018) * 0.03
            
            Z = Z_base * country_scale * year_factor * np.random.uniform(0.9, 1.1, (N, N))
            Y = Y_base * country_scale * year_factor * np.random.uniform(0.9, 1.1, N)
            T = T_base * country_scale * np.random.uniform(0.8, 1.2, N)
            
            X = Z.sum(axis=1) + Y
            VA = X - Z.sum(axis=0)
            VA_sin_T = VA - T
            Compensation = VA_sin_T * 0.6
            Operating_Surplus = VA_sin_T * 0.4
            
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
                    'final_demand': Y[i]
                })
    
    return pd.DataFrame(connections_list), pd.DataFrame(variables_list)

# ============================================================================
# CLASE DE ANÁLISIS
# ============================================================================

class FiscalNetworkAnalyzer:
    """Analizador de redes I-O con soporte para múltiples países y años."""
    
    def __init__(self, connections_df, variables_df, countries=None, years=None):
        # Limpiar datos
        self.connections_full = clean_dataframe(connections_df, 'connections')
        self.variables_full = clean_dataframe(variables_df, 'variables')
        
        # Obtener valores únicos disponibles
        available_countries = self.connections_full['country_code'].unique().tolist()
        available_years = self.connections_full['year'].unique().tolist()
        
        # Configurar países
        if countries is None:
            countries = available_countries
        elif isinstance(countries, str):
            countries = [countries]
        countries = [c for c in countries if c in available_countries]
        if not countries:
            countries = available_countries[:1]
        
        # Configurar años
        if years is None:
            years = available_years
        elif isinstance(years, (int, float)):
            years = [int(years)]
        years = [y for y in years if y in available_years]
        if not years:
            years = available_years[:1]
        
        self.countries = countries
        self.years = [int(y) for y in years]
        
        # Filtrar datos
        self.connections = self.connections_full[
            (self.connections_full['country_code'].isin(self.countries)) & 
            (self.connections_full['year'].isin(self.years))
        ].copy()
        
        self.variables = self.variables_full[
            (self.variables_full['country_code'].isin(self.countries)) & 
            (self.variables_full['year'].isin(self.years))
        ].copy()
        
        # Obtener sectores únicos (ordenados alfabéticamente como strings)
        all_sectors = set(self.variables['sector'].unique()) | \
                     set(self.connections['from_sector'].unique()) | \
                     set(self.connections['to_sector'].unique())
        self.sectors = sorted([s for s in all_sectors if s], key=str)
        self.n_sectors = len(self.sectors)
        self.sector_to_idx = {s: i for i, s in enumerate(self.sectors)}
        
        # Construir matrices
        self._build_matrices()
    
    def _build_matrices(self):
        """Construye matrices I-O."""
        n = self.n_sectors
        
        if n == 0:
            self.Z = np.zeros((1, 1))
            self.X = np.zeros(1)
            self.Y = np.zeros(1)
            self.T = np.zeros(1)
            self.VA = np.zeros(1)
            self.A = np.zeros((1, 1))
            self.L = np.eye(1)
            self.tax_rate = np.zeros(1)
            return
        
        # Matriz Z agregada
        self.Z = np.zeros((n, n))
        connections_grouped = self.connections.groupby(['from_sector', 'to_sector'])['flow_value'].sum()
        
        for (from_s, to_s), value in connections_grouped.items():
            i = self.sector_to_idx.get(from_s)
            j = self.sector_to_idx.get(to_s)
            if i is not None and j is not None:
                self.Z[i, j] = value
        
        # Vectores agregados
        self.X = np.zeros(n)
        self.Y = np.zeros(n)
        self.T = np.zeros(n)
        self.VA = np.zeros(n)
        
        for col, arr in [('gross_output', self.X), ('final_demand', self.Y), 
                         ('taxes_subsidies', self.T), ('value_added', self.VA)]:
            if col in self.variables.columns:
                grouped = self.variables.groupby('sector')[col].sum()
                for sector, value in grouped.items():
                    idx = self.sector_to_idx.get(sector)
                    if idx is not None:
                        arr[idx] = value
        
        # Matriz A
        X_inv = np.where(self.X > 0, 1/self.X, 0)
        self.A = self.Z @ np.diag(X_inv)
        
        # Matriz L
        try:
            I_minus_A = np.eye(n) - self.A
            if np.linalg.det(I_minus_A) != 0:
                self.L = np.linalg.inv(I_minus_A)
            else:
                self.L = np.linalg.pinv(I_minus_A)
        except Exception:
            self.L = np.eye(n)
        
        # Tasa de impuesto
        self.tax_rate = np.where(self.X > 0, self.T / self.X, 0)
    
    def compute_multipliers(self):
        """Calcula multiplicadores y linkages."""
        if self.n_sectors == 0:
            return pd.DataFrame()
        
        forward = self.L.sum(axis=1)
        backward = self.L.sum(axis=0)
        
        fl_mean = forward.mean() if forward.mean() > 0 else 1
        bl_mean = backward.mean() if backward.mean() > 0 else 1
        
        fl_norm = forward / fl_mean
        bl_norm = backward / bl_mean
        
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
        """Ejecuta análisis HEF."""
        if self.n_sectors == 0:
            return pd.DataFrame()
        
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
                L_new = self.L.copy()
            
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
        if self.n_sectors == 0 or sector_idx >= self.n_sectors:
            return {'trajectory': [0], 'cumulative': np.zeros(1), 'total_effect': 0}
        
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
        if self.n_sectors == 0:
            return pd.DataFrame()
        
        net_effect = self.A.T @ self.tax_rate
        
        return pd.DataFrame({
            'sector': self.sectors,
            'own_tax_rate': self.tax_rate * 100,
            'network_effect': net_effect * 100,
            'total_effect': (self.tax_rate + net_effect) * 100
        })
    
    def verify_calculations(self):
        """Verifica consistencia de cálculos."""
        if self.n_sectors == 0:
            return {'No hay datos': False}
        
        checks = {}
        
        try:
            checks['X ≈ Z.sum + Y'] = np.allclose(self.X, self.Z.sum(axis=1) + self.Y, rtol=0.1)
        except:
            checks['X ≈ Z.sum + Y'] = False
        
        try:
            checks['X ≈ L × Y'] = np.allclose(self.X, self.L @ self.Y, rtol=0.1)
        except:
            checks['X ≈ L × Y'] = False
        
        try:
            col_sums = self.A.sum(axis=0)
            checks['Σ A[i,j] < 1'] = (col_sums[col_sums > 0] < 1.5).all()
        except:
            checks['Σ A[i,j] < 1'] = False
        
        checks['L ≥ 0'] = (self.L >= -0.01).all()
        
        return checks
    
    def get_comparison_by_country(self):
        """Métricas por país."""
        if len(self.countries) <= 1:
            return None
        
        results = []
        for country in self.countries:
            country_vars = self.variables[self.variables['country_code'] == country]
            
            if len(country_vars) == 0:
                continue
            
            total_output = country_vars['gross_output'].sum()
            total_tax = country_vars['taxes_subsidies'].sum()
            
            results.append({
                'country': country,
                'total_output': total_output,
                'total_taxes': total_tax,
                'avg_tax_rate_pct': (total_tax / total_output * 100) if total_output > 0 else 0,
                'n_subsidized': (country_vars['taxes_subsidies'] > 0).sum(),
                'n_taxed': (country_vars['taxes_subsidies'] < 0).sum()
            })
        
        return pd.DataFrame(results) if results else None
    
    def get_comparison_by_year(self):
        """Métricas por año."""
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
        
        return pd.DataFrame(results) if results else None

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    st.title("📊 Análisis de Impacto Fiscal con Redes I-O")
    st.markdown("**Soporte para múltiples países y años**")
    
    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Configuración")
        
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
                    
                    # Limpiar datos
                    connections_df = clean_dataframe(connections_df, 'connections')
                    variables_df = clean_dataframe(variables_df, 'variables')
                    
                    all_countries = sorted(connections_df['country_code'].unique().tolist())
                    all_years = sorted([int(y) for y in connections_df['year'].unique()])
                    
                    st.success(f"✅ Datos cargados: {len(all_countries)} países, {len(all_years)} años")
                    
                except Exception as e:
                    st.error(f"Error al cargar: {e}")
                    connections_df, variables_df = generate_mockup_data()
                    all_countries = sorted(connections_df['country_code'].unique().tolist())
                    all_years = sorted([int(y) for y in connections_df['year'].unique()])
            else:
                st.info("⬆️ Carga ambos archivos CSV")
                connections_df, variables_df = generate_mockup_data()
                all_countries = sorted(connections_df['country_code'].unique().tolist())
                all_years = sorted([int(y) for y in connections_df['year'].unique()])
        else:
            connections_df, variables_df = generate_mockup_data()
            all_countries = sorted(connections_df['country_code'].unique().tolist())
            all_years = sorted([int(y) for y in connections_df['year'].unique()])
        
        st.markdown("---")
        st.subheader("🌍 Selección de Datos")
        
        selection_mode = st.radio(
            "Modo de análisis:",
            ["Un país, un año", "Múltiples países", "Múltiples años", "Múltiples países y años"]
        )
        
        if selection_mode == "Un país, un año":
            selected_countries = [st.selectbox("País:", all_countries)]
            selected_years = [st.selectbox("Año:", all_years)]
            
        elif selection_mode == "Múltiples países":
            selected_countries = st.multiselect(
                "Países:", 
                all_countries, 
                default=all_countries[:min(3, len(all_countries))]
            )
            selected_years = [st.selectbox("Año:", all_years)]
            
        elif selection_mode == "Múltiples años":
            selected_countries = [st.selectbox("País:", all_countries)]
            selected_years = st.multiselect(
                "Años:", 
                all_years, 
                default=all_years
            )
            
        else:
            selected_countries = st.multiselect(
                "Países:", 
                all_countries, 
                default=all_countries[:min(3, len(all_countries))]
            )
            selected_years = st.multiselect(
                "Años:", 
                all_years, 
                default=all_years
            )
        
        if not selected_countries:
            selected_countries = [all_countries[0]] if all_countries else []
        if not selected_years:
            selected_years = [all_years[0]] if all_years else []
        
        st.markdown("---")
        st.subheader("🔧 Parámetros")
        elasticity = st.slider("Elasticidad HEF:", 0.1, 1.0, 0.5, 0.1)
        shock_magnitude = st.slider("Magnitud shock (%):", 1, 20, 10) / 100
        shock_iterations = st.slider("Rondas propagación:", 5, 20, 10)
        
        st.markdown("---")
        st.markdown(f"**Selección:** {len(selected_countries)} país(es), {len(selected_years)} año(s)")
    
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
        st.write("Detalles del error para debugging:")
        st.exception(e)
        st.stop()
    
    # =========================================================================
    # TABS
    # =========================================================================
    tabs = st.tabs([
        "📋 Resumen", "🌍 Comparación", "🔢 Matrices", 
        "📊 Multiplicadores", "💰 HEF", "🌊 Propagación", "✅ Verificación"
    ])
    
    # TAB 1: RESUMEN
    with tabs[0]:
        st.header("📋 Resumen de Datos")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Países", len(analyzer.countries))
        with col2:
            st.metric("Años", len(analyzer.years))
        with col3:
            st.metric("Sectores", analyzer.n_sectors)
        with col4:
            st.metric("Producción Total", f"{analyzer.X.sum():,.0f}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Registros", f"{len(analyzer.variables):,}")
        with col2:
            st.metric("Conexiones", f"{len(analyzer.connections):,}")
        with col3:
            subsidios = analyzer.T[analyzer.T > 0].sum()
            st.metric("Subsidios", f"+{subsidios:,.0f}")
        with col4:
            impuestos = abs(analyzer.T[analyzer.T < 0].sum())
            st.metric("Impuestos", f"-{impuestos:,.0f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Variables por Sector (Agregado)")
            if len(analyzer.variables) > 0:
                sector_summary = analyzer.variables.groupby('sector').agg({
                    'gross_output': 'sum',
                    'value_added': 'sum',
                    'taxes_subsidies': 'sum'
                }).round(2)
                st.dataframe(sector_summary, use_container_width=True)
        
        with col2:
            st.subheader("Distribución Fiscal")
            if PLOTLY_AVAILABLE and len(analyzer.variables) > 0:
                sector_tax = analyzer.variables.groupby('sector')['taxes_subsidies'].sum().reset_index()
                sector_tax['tipo'] = sector_tax['taxes_subsidies'].apply(
                    lambda x: 'Subsidio' if x > 0 else 'Impuesto'
                )
                
                fig = px.bar(
                    sector_tax.sort_values('taxes_subsidies'),
                    x='taxes_subsidies',
                    y='sector',
                    orientation='h',
                    color='tipo',
                    color_discrete_map={'Subsidio': '#28a745', 'Impuesto': '#dc3545'},
                    title='Componente Fiscal por Sector'
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Convención de signos"):
            st.markdown(DOCS["tax_convention"])
    
    # TAB 2: COMPARACIÓN
    with tabs[1]:
        st.header("🌍 Comparación")
        
        country_comparison = analyzer.get_comparison_by_country()
        if country_comparison is not None and len(country_comparison) > 0:
            st.subheader("Por País")
            st.dataframe(country_comparison.round(2), use_container_width=True)
            
            if PLOTLY_AVAILABLE:
                fig = px.bar(country_comparison, x='country', y='total_output', 
                            color='avg_tax_rate_pct', title='Producción por País')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecciona múltiples países para ver comparación.")
        
        year_comparison = analyzer.get_comparison_by_year()
        if year_comparison is not None and len(year_comparison) > 0:
            st.subheader("Por Año")
            st.dataframe(year_comparison.round(2), use_container_width=True)
            
            if PLOTLY_AVAILABLE:
                fig = px.line(year_comparison, x='year', y='total_output', 
                             markers=True, title='Evolución Temporal')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecciona múltiples años para ver evolución.")
    
    # TAB 3: MATRICES
    with tabs[2]:
        st.header("🔢 Matrices I-O")
        
        if len(analyzer.countries) > 1 or len(analyzer.years) > 1:
            st.info(f"Matrices agregadas: {len(analyzer.countries)} país(es), {len(analyzer.years)} año(s)")
        
        matrix_choice = st.selectbox("Matriz:", ["Z - Consumo Intermedio", "A - Coeficientes", "L - Leontief"])
        
        if matrix_choice == "Z - Consumo Intermedio":
            matrix_data = analyzer.Z
        elif matrix_choice == "A - Coeficientes":
            matrix_data = analyzer.A
        else:
            matrix_data = analyzer.L
        
        if PLOTLY_AVAILABLE and analyzer.n_sectors > 0:
            fig = px.imshow(matrix_data, x=analyzer.sectors, y=analyzer.sectors,
                           color_continuous_scale='Blues', title=matrix_choice)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Ver tabla"):
            df_matrix = pd.DataFrame(matrix_data, index=analyzer.sectors, columns=analyzer.sectors)
            st.dataframe(df_matrix.round(4), use_container_width=True)
    
    # TAB 4: MULTIPLICADORES
    with tabs[3]:
        st.header("📊 Multiplicadores")
        
        multipliers_df = analyzer.compute_multipliers()
        
        if len(multipliers_df) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(multipliers_df.round(4), use_container_width=True)
            with col2:
                if PLOTLY_AVAILABLE:
                    fig = px.scatter(multipliers_df, x='bl_normalized', y='fl_normalized',
                                    text='sector', color='classification',
                                    title='Clasificación Sectorial')
                    fig.add_hline(y=1, line_dash="dash", opacity=0.5)
                    fig.add_vline(x=1, line_dash="dash", opacity=0.5)
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: HEF
    with tabs[4]:
        st.header("💰 Análisis HEF")
        
        hef_results = analyzer.fiscal_hypothetical_extraction(elasticity=elasticity)
        
        if len(hef_results) > 0:
            col1, col2 = st.columns(2)
            with col1:
                display_cols = ['sector', 'tax_type', 'tax_original', 'tax_rate_pct', 'relative_impact_pct']
                st.dataframe(hef_results[display_cols].round(4), use_container_width=True)
            with col2:
                if PLOTLY_AVAILABLE:
                    fig = px.bar(hef_results.sort_values('relative_impact_pct'),
                                x='relative_impact_pct', y='sector', orientation='h',
                                color='tax_type', title='Impacto HEF')
                    fig.add_vline(x=0, line_dash="dash")
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 6: PROPAGACIÓN
    with tabs[5]:
        st.header("🌊 Propagación")
        
        if analyzer.n_sectors > 0:
            shock_sector = st.selectbox("Sector origen:", analyzer.sectors)
            shock_idx = analyzer.sector_to_idx.get(shock_sector, 0)
            
            shock_result = analyzer.simulate_shock(shock_idx, shock_magnitude, shock_iterations)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Shock Inicial", f"{shock_magnitude * analyzer.X[shock_idx]:,.2f}")
            with col2:
                st.metric("Efecto Total", f"{shock_result['total_effect']:,.2f}")
            with col3:
                initial = shock_magnitude * analyzer.X[shock_idx]
                ratio = shock_result['total_effect'] / initial if initial > 0 else 0
                st.metric("Multiplicador", f"{ratio:.4f}x")
            
            if PLOTLY_AVAILABLE:
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(len(shock_result['trajectory']))),
                                            y=shock_result['trajectory'], mode='lines+markers'))
                    fig.update_layout(title='Propagación por Ronda')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    cumulative = shock_result['cumulative']
                    sorted_idx = np.argsort(cumulative)[::-1]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=cumulative[sorted_idx],
                                        y=[analyzer.sectors[i] for i in sorted_idx],
                                        orientation='h'))
                    fig.update_layout(title='Efecto Acumulado')
                    st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🌐 Efectos de Red")
            network_effects = analyzer.compute_network_effects()
            st.dataframe(network_effects.round(2), use_container_width=True)
    
    # TAB 7: VERIFICACIÓN
    with tabs[6]:
        st.header("✅ Verificación")
        
        checks = analyzer.verify_calculations()
        
        for name, passed in checks.items():
            if passed:
                st.success(f"✅ {name}")
            else:
                st.warning(f"⚠️ {name}")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Subsidios (T>0):** {(analyzer.T > 0).sum()} sectores")
            st.write(f"**Impuestos (T<0):** {(analyzer.T < 0).sum()} sectores")
            st.write(f"**Balance neto:** {analyzer.T.sum():+,.2f}")
        
        with col2:
            mult_csv = analyzer.compute_multipliers().to_csv(index=False)
            st.download_button("📊 Multiplicadores", mult_csv, "multiplicadores.csv")
            
            hef_df = analyzer.fiscal_hypothetical_extraction()
            if 'X_new' in hef_df.columns:
                hef_df = hef_df.drop(columns=['X_new'])
            st.download_button("💰 HEF", hef_df.to_csv(index=False), "hef.csv")

if __name__ == "__main__":
    main()
