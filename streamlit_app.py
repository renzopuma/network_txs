"""
📊 Dashboard de Análisis Fiscal con Enfoque de Redes
====================================================
Versión completa con documentación y fórmulas integradas

Ejecutar con: streamlit run fiscal_dashboard_v3.py
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
# IMPORTAR PLOTLY CON MANEJO DE ERRORES
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
# TEXTOS DE DOCUMENTACIÓN Y FÓRMULAS
# ============================================================================

DOCS = {
    "intro": """
    ## 📊 Análisis del Impacto Fiscal con Enfoque de Redes
    
    Este dashboard implementa metodologías de **Input-Output** para analizar cómo los impuestos 
    y subsidios afectan la producción sectorial, considerando las **interdependencias entre sectores**.
    
    ### 🎯 Objetivo
    Medir el impacto sistémico de la política fiscal considerando que los sectores económicos 
    están conectados a través de cadenas de suministro.
    
    ### 📚 Referencia Metodológica
    - Miller & Blair (2009): "Input-Output Analysis: Foundations and Extensions"
    - Base de datos EORA26: https://worldmrio.com/
    """,
    
    "tax_convention": """
    ### ⚠️ Convención del Campo `taxes_subsidies`
    
    | Signo | Interpretación | Ejemplo |
    |-------|---------------|---------|
    | **T > 0** | 🟢 **SUBSIDIO NETO** - El sector recibe transferencias | Agricultura, Transporte público |
    | **T < 0** | 🔴 **IMPUESTO NETO** - El sector paga al gobierno | Minería, Manufactura, Servicios financieros |
    
    **Sectores típicamente subsidiados:** Agricultura, Pesca, Educación y Salud, Transporte público
    
    **Sectores típicamente gravados:** Minería, Petróleo y Químicos, Manufactura, Servicios financieros
    """,
    
    "matrix_z": """
    ### 📦 Matriz Z - Consumo Intermedio
    
    La matriz **Z** representa los flujos monetarios entre sectores.
    
    **Interpretación:** `Z[i,j]` = Cuánto compra el sector `j` del sector `i` (en millones USD)
    
    - Las **filas** representan las ventas de cada sector (proveedores)
    - Las **columnas** representan las compras de cada sector (demandantes)
    - La **diagonal** representa transacciones intra-sectoriales
    """,
    
    "matrix_a": """
    ### 📐 Matriz A - Coeficientes Técnicos
    
    **Fórmula:**
    ```
    A = Z × diag(X)⁻¹
    ```
    
    **Donde:**
    - `Z`: Matriz de consumo intermedio (n×n)
    - `X`: Vector de producción bruta (n×1)
    
    **Interpretación:** `A[i,j]` = Cantidad de insumo del sector `i` necesario para producir **1 unidad** del sector `j`
    
    **Propiedad importante:** La suma de cada columna debe ser < 1 para que la economía sea viable 
    (debe quedar margen para el valor agregado).
    """,
    
    "matrix_l": """
    ### 🔄 Matriz L - Leontief (Multiplicadores)
    
    **Fórmula:**
    ```
    L = (I - A)⁻¹
    ```
    
    **Donde:**
    - `I`: Matriz identidad
    - `A`: Matriz de coeficientes técnicos
    
    **Interpretación:** `L[i,j]` = Producción **total** del sector `i` necesaria (directa + indirecta) 
    para satisfacer **1 unidad** de demanda final del sector `j`
    
    **Propiedades:**
    - Todos los elementos son ≥ 0
    - La diagonal es siempre ≥ 1 (incluye el efecto directo)
    - Cumple la identidad: `X = L × Y`
    """,
    
    "multipliers": """
    ### 📊 Multiplicadores y Linkages
    
    #### Multiplicador Tipo I
    ```
    M[j] = Σᵢ L[i,j]  (suma de la columna j)
    ```
    **Interpretación:** Producción total generada en **toda la economía** por cada unidad de demanda final del sector `j`.
    
    #### Forward Linkage (FL)
    ```
    FL[i] = Σⱼ L[i,j]  (suma de la fila i)
    ```
    **Interpretación:** Importancia del sector `i` como **proveedor** de insumos a otros sectores.
    
    #### Backward Linkage (BL)
    ```
    BL[j] = Σᵢ L[i,j]  (suma de la columna j)
    ```
    **Interpretación:** Importancia del sector `j` como **demandante** de insumos de otros sectores.
    
    #### Clasificación Sectorial
    | FL Normalizado | BL Normalizado | Clasificación |
    |----------------|----------------|---------------|
    | > 1 | > 1 | 🔴 **Sector Clave** - Alto impacto como proveedor Y demandante |
    | > 1 | ≤ 1 | 🔵 **Forward Oriented** - Importante proveedor de insumos |
    | ≤ 1 | > 1 | 🟢 **Backward Oriented** - Importante demandante de insumos |
    | ≤ 1 | ≤ 1 | ⚪ **Linkages Débiles** - Poco integrado en la economía |
    """,
    
    "hef_method": """
    ### 🔬 Método de Extracción Hipotética Fiscal (HEF)
    
    **Objetivo:** Medir la **importancia sistémica** del componente fiscal de cada sector.
    
    #### Mecanismo
    1. **Calcular** producción con el impuesto/subsidio actual
    2. **Simular** eliminación del componente fiscal → ajuste en costos
    3. **Recalcular** equilibrio de producción (nueva matriz L)
    4. **Medir** el cambio en la producción total
    
    #### Lógica de Signos
    | Acción | Efecto en Costos | Efecto en Producción |
    |--------|------------------|----------------------|
    | Eliminar **IMPUESTO** (T<0) | Costos **BAJAN** | Producción **SUBE** ↑ |
    | Eliminar **SUBSIDIO** (T>0) | Costos **SUBEN** | Producción **BAJA** ↓ |
    
    #### Fórmula del Ajuste
    ```
    Cambio en costo = -t[j]  (negativo de la tasa fiscal)
    
    A_nuevo[j,i] = A[j,i] × (1 + elasticidad × cambio_costo)
    ```
    
    #### Métricas de Resultado
    - **Impacto Absoluto:** `ΔX_total = X_nuevo - X_original`
    - **Impacto Relativo:** `ΔX% = (ΔX_total / X_original) × 100`
    - **Importancia Sistémica:** Ratio de efecto indirecto vs directo
    """,
    
    "tax_rate": """
    ### 💰 Tasa de Impuesto por Unidad
    
    **Fórmula:**
    ```
    t[i] = T[i] / X[i]
    ```
    
    **Donde:**
    - `T[i]`: Impuesto neto (-) o subsidio neto (+) del sector `i`
    - `X[i]`: Producción bruta del sector `i`
    
    **Interpretación:** Proporción del componente fiscal respecto a la producción total del sector.
    """,
    
    "network_effects": """
    ### 🌐 Efecto de Red (Spillover Fiscal)
    
    **Fórmula:**
    ```
    NetEffect[j] = Σᵢ A[i,j] × t[i]
    ```
    
    **Interpretación:** Impacto ponderado de los impuestos/subsidios de los **proveedores** del sector `j`.
    
    - **NetEffect > 0:** El sector se beneficia de subsidios a sus proveedores (menores costos de insumos)
    - **NetEffect < 0:** El sector sufre por impuestos de sus proveedores (mayores costos de insumos)
    
    **Efecto Total:**
    ```
    Efecto_Total[j] = t[j] + NetEffect[j]
    ```
    Combina el efecto fiscal propio más el efecto indirecto de la red.
    """,
    
    "shock_propagation": """
    ### 🌊 Propagación de Shocks
    
    **Mecanismo:** Un shock inicial en un sector se propaga a través de la red en rondas sucesivas.
    
    **Fórmula iterativa:**
    ```
    ε⁽⁰⁾ = shock inicial (solo en sector j)
    ε⁽ᵏ⁺¹⁾ = Aᵀ × ε⁽ᵏ⁾
    
    Efecto acumulado = Σₖ ε⁽ᵏ⁾
    ```
    
    **Convergencia:** El efecto acumulado converge a `(I - Aᵀ)⁻¹ × ε⁽⁰⁾ = Lᵀ × ε⁽⁰⁾`
    
    **Ratio Multiplicador:** `Efecto_Total / Shock_Inicial` indica cuántas veces se amplifica el shock.
    """,
    
    "eora26_sectors": """
    ### 🏭 Sectores EORA26
    
    | # | Sector | # | Sector |
    |---|--------|---|--------|
    | 1 | Agriculture | 14 | Construction |
    | 2 | Fishing | 15 | Maintenance and Repair |
    | 3 | Mining and Quarrying | 16 | Wholesale Trade |
    | 4 | Food & Beverages | 17 | Retail Trade |
    | 5 | Textiles and Wearing Apparel | 18 | Hotels and Restaurants |
    | 6 | Wood and Paper | 19 | Transport |
    | 7 | Petroleum, Chemical Products | 20 | Post and Telecommunications |
    | 8 | Metal Products | 21 | Financial Services |
    | 9 | Electrical and Machinery | 22 | Public Administration |
    | 10 | Transport Equipment | 23 | Education, Health and Other Services |
    | 11 | Other Manufacturing | 24 | Private Households |
    | 12 | Recycling | 25 | Others |
    | 13 | Electricity, Gas and Water | 26 | Re-export & Re-import |
    """
}

# ============================================================================
# FUNCIONES DE GENERACIÓN DE DATOS MOCKUP
# ============================================================================

def generate_mockup_data():
    """Genera datos de ejemplo con 8 sectores."""
    
    SECTORS = [
        'Agricultura', 'Minería', 'Manufactura', 'Energía',
        'Construcción', 'Comercio', 'Transporte', 'Servicios'
    ]
    N = len(SECTORS)
    
    # Matriz Z de consumo intermedio
    Z = np.array([
        [20, 5, 80, 2, 3, 10, 5, 5],
        [2, 30, 60, 15, 40, 2, 5, 3],
        [15, 10, 100, 20, 80, 30, 25, 40],
        [8, 25, 40, 10, 15, 20, 30, 25],
        [3, 5, 10, 5, 20, 8, 10, 15],
        [5, 3, 15, 3, 10, 15, 12, 20],
        [10, 15, 30, 8, 25, 18, 10, 15],
        [7, 12, 25, 12, 20, 25, 18, 30],
    ], dtype=float)
    
    # Demanda final
    Y = np.array([150, 80, 400, 120, 200, 250, 100, 300], dtype=float)
    
    # Producción bruta
    X = Z.sum(axis=1) + Y
    
    # Valor agregado
    VA = X - Z.sum(axis=0)
    
    # Impuestos/Subsidios (T > 0 = subsidio, T < 0 = impuesto)
    T = np.array([+15.0, -25.0, -45.0, -10.0, -30.0, -20.0, +8.0, -35.0])
    
    # Componentes del VA
    VA_sin_T = VA - T
    Compensation = VA_sin_T * 0.6
    Operating_Surplus = VA_sin_T * 0.4
    
    # Crear DataFrames
    connections_list = []
    for i in range(N):
        for j in range(N):
            if Z[i, j] > 0:
                connections_list.append({
                    'country_code': 'TEST',
                    'year': 2020,
                    'from_sector': SECTORS[i],
                    'to_sector': SECTORS[j],
                    'flow_value': Z[i, j]
                })
    
    variables_list = []
    for i in range(N):
        variables_list.append({
            'country_code': 'TEST',
            'year': 2020,
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
    
    return pd.DataFrame(connections_list), pd.DataFrame(variables_list), SECTORS

# ============================================================================
# CLASE PRINCIPAL DE ANÁLISIS
# ============================================================================

class FiscalNetworkAnalyzer:
    """Clase para análisis de redes Input-Output con enfoque fiscal."""
    
    def __init__(self, connections_df, variables_df, country='TEST', year=2020):
        self.country = country
        self.year = year
        
        # Filtrar datos
        self.connections = connections_df[
            (connections_df['country_code'] == country) & 
            (connections_df['year'] == year)
        ].copy()
        
        self.variables = variables_df[
            (variables_df['country_code'] == country) & 
            (variables_df['year'] == year)
        ].copy()
        
        # Sectores
        self.sectors = sorted(self.variables['sector'].unique())
        self.n_sectors = len(self.sectors)
        self.sector_to_idx = {s: i for i, s in enumerate(self.sectors)}
        
        # Construir matrices
        self._build_matrices()
    
    def _build_matrices(self):
        """Construye matrices I-O fundamentales."""
        n = self.n_sectors
        
        # Matriz Z
        self.Z = np.zeros((n, n))
        for _, row in self.connections.iterrows():
            i = self.sector_to_idx.get(row['from_sector'])
            j = self.sector_to_idx.get(row['to_sector'])
            if i is not None and j is not None:
                self.Z[i, j] = row['flow_value']
        
        # Vectores
        self.X = np.zeros(n)
        self.Y = np.zeros(n)
        self.T = np.zeros(n)
        self.VA = np.zeros(n)
        
        for _, row in self.variables.iterrows():
            idx = self.sector_to_idx.get(row['sector'])
            if idx is not None:
                self.X[idx] = row['gross_output']
                self.Y[idx] = row['final_demand']
                self.T[idx] = row['taxes_subsidies']
                self.VA[idx] = row['value_added']
        
        # Matriz A = Z × diag(X)⁻¹
        X_inv = np.where(self.X > 0, 1/self.X, 0)
        self.A = self.Z @ np.diag(X_inv)
        
        # Matriz L = (I - A)⁻¹
        try:
            self.L = np.linalg.inv(np.eye(n) - self.A)
        except Exception:
            self.L = np.linalg.pinv(np.eye(n) - self.A)
        
        # Tasa de impuesto: t = T / X
        self.tax_rate = np.where(self.X > 0, self.T / self.X, 0)
    
    def compute_multipliers(self):
        """Calcula multiplicadores y linkages."""
        forward = self.L.sum(axis=1)  # Suma por fila
        backward = self.L.sum(axis=0)  # Suma por columna
        
        fl_norm = forward / forward.mean()
        bl_norm = backward / backward.mean()
        
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
        """
        Ejecuta análisis HEF para todos los sectores.
        
        Mecanismo:
        1. Si T[j] < 0 (impuesto): eliminarlo REDUCE costos → más producción
        2. Si T[j] > 0 (subsidio): eliminarlo AUMENTA costos → menos producción
        """
        results = []
        X_total_original = self.X.sum()
        
        for j in range(self.n_sectors):
            # Cambio en costos al eliminar el componente fiscal
            # cost_change = -tax_rate (si era impuesto negativo, costos bajan)
            cost_change = -self.tax_rate[j]
            
            # Ajustar matriz A
            A_new = self.A.copy()
            for i in range(self.n_sectors):
                if self.A[j, i] > 0:
                    demand_change = elasticity * cost_change
                    A_new[j, i] = self.A[j, i] * (1 + demand_change)
            
            # Recalcular L y X
            try:
                L_new = np.linalg.inv(np.eye(self.n_sectors) - A_new)
            except Exception:
                L_new = np.linalg.pinv(np.eye(self.n_sectors) - A_new)
            
            X_new = L_new @ self.Y
            X_total_new = X_new.sum()
            
            absolute_impact = X_total_new - X_total_original
            relative_impact = (absolute_impact / X_total_original) * 100
            
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
        """
        Simula propagación de shock fiscal.
        
        ε⁽⁰⁾ = shock inicial
        ε⁽ᵏ⁺¹⁾ = Aᵀ × ε⁽ᵏ⁾
        """
        epsilon = np.zeros(self.n_sectors)
        epsilon[sector_idx] = magnitude * self.X[sector_idx]
        
        W = self.A.T  # Matriz de transmisión
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
        """
        Calcula efectos de red (spillovers fiscales).
        
        NetEffect[j] = Σᵢ A[i,j] × t[i]
        """
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
            'VA = X - Z.sum(col)': np.allclose(self.VA, self.X - self.Z.sum(axis=0)),
            'X = L × Y': np.allclose(self.X, self.L @ self.Y),
            'Σ A[i,j] < 1 ∀j': (self.A.sum(axis=0) < 1).all(),
            'L ≥ 0': (self.L >= 0).all(),
            'L diagonal ≥ 1': (np.diag(self.L) >= 1).all()
        }
        return checks

# ============================================================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# ============================================================================

def main():
    # Header
    st.title("📊 Análisis de Impacto Fiscal con Redes I-O")
    st.markdown("**Métodos de Extracción Hipotética Fiscal y Framework Integrado**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Fuente de datos
        data_source = st.radio(
            "Fuente de datos:",
            ["Datos de ejemplo (8 sectores)", "Cargar mis datos CSV"]
        )
        
        if data_source == "Cargar mis datos CSV":
            st.markdown("---")
            st.subheader("Cargar archivos")
            
            connections_file = st.file_uploader("connections.csv", type=['csv'])
            variables_file = st.file_uploader("sector_variables.csv", type=['csv'])
            
            if connections_file and variables_file:
                try:
                    connections_df = pd.read_csv(connections_file)
                    variables_df = pd.read_csv(variables_file)
                    
                    countries = connections_df['country_code'].unique().tolist()
                    years = connections_df['year'].unique().tolist()
                    
                    selected_country = st.selectbox("País:", countries)
                    selected_year = st.selectbox("Año:", years)
                except Exception as e:
                    st.error(f"Error al cargar archivos: {e}")
                    connections_df, variables_df, _ = generate_mockup_data()
                    selected_country = 'TEST'
                    selected_year = 2020
            else:
                st.info("⬆️ Carga ambos archivos CSV")
                connections_df, variables_df, _ = generate_mockup_data()
                selected_country = 'TEST'
                selected_year = 2020
        else:
            connections_df, variables_df, _ = generate_mockup_data()
            selected_country = 'TEST'
            selected_year = 2020
        
        st.markdown("---")
        st.subheader("Parámetros HEF")
        elasticity = st.slider(
            "Elasticidad precio:", 
            0.1, 1.0, 0.5, 0.1,
            help="Elasticidad de la demanda respecto al precio. Mayor elasticidad = mayor sensibilidad a cambios de costos."
        )
        
        st.markdown("---")
        st.subheader("Simulación de Shock")
        shock_magnitude = st.slider(
            "Magnitud del shock (%):", 
            1, 20, 10,
            help="Porcentaje del output del sector que representa el shock inicial."
        ) / 100
        shock_iterations = st.slider(
            "Rondas de propagación:", 
            5, 20, 10,
            help="Número de iteraciones para simular la propagación del shock."
        )
    
    # Crear analizador
    try:
        analyzer = FiscalNetworkAnalyzer(connections_df, variables_df, selected_country, selected_year)
    except Exception as e:
        st.error(f"Error al crear el analizador: {e}")
        st.stop()
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📚 Documentación", "📋 Datos", "🔢 Matrices I-O", 
        "📊 Multiplicadores", "💰 Análisis HEF", "🌊 Propagación", "✅ Verificación"
    ])
    
    # ==========================================================================
    # TAB 1: DOCUMENTACIÓN
    # ==========================================================================
    with tab1:
        st.markdown(DOCS["intro"])
        
        st.markdown("---")
        st.markdown(DOCS["tax_convention"])
        
        st.markdown("---")
        with st.expander("📐 Fórmulas Matemáticas - Matrices", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(DOCS["matrix_a"])
            with col2:
                st.markdown(DOCS["matrix_l"])
        
        with st.expander("📊 Fórmulas - Multiplicadores y Linkages", expanded=False):
            st.markdown(DOCS["multipliers"])
        
        with st.expander("🔬 Método HEF - Extracción Hipotética Fiscal", expanded=False):
            st.markdown(DOCS["hef_method"])
        
        with st.expander("🌐 Efectos de Red y Propagación", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(DOCS["network_effects"])
            with col2:
                st.markdown(DOCS["shock_propagation"])
        
        with st.expander("🏭 Sectores EORA26", expanded=False):
            st.markdown(DOCS["eora26_sectors"])
    
    # ==========================================================================
    # TAB 2: DATOS
    # ==========================================================================
    with tab2:
        st.header("📋 Datos de Entrada")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sectores", analyzer.n_sectors)
        with col2:
            st.metric("Producción Total", f"{analyzer.X.sum():,.0f} M")
        with col3:
            subsidios = analyzer.T[analyzer.T > 0].sum()
            st.metric("Total Subsidios", f"+{subsidios:,.0f} M")
        with col4:
            impuestos = abs(analyzer.T[analyzer.T < 0].sum())
            st.metric("Total Impuestos", f"-{impuestos:,.0f} M")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Variables por Sector")
            display_df = variables_df[variables_df['country_code'] == selected_country][
                ['sector', 'gross_output', 'value_added', 'taxes_subsidies', 'final_demand']
            ].copy()
            st.dataframe(display_df, use_container_width=True)
        
        with col2:
            st.subheader("Distribución Fiscal")
            if PLOTLY_AVAILABLE:
                df_plot = variables_df[variables_df['country_code'] == selected_country].copy()
                df_plot['tipo'] = df_plot['taxes_subsidies'].apply(lambda x: 'Subsidio' if x > 0 else 'Impuesto')
                
                fig = px.bar(
                    df_plot.sort_values('taxes_subsidies'),
                    x='taxes_subsidies',
                    y='sector',
                    orientation='h',
                    color='tipo',
                    color_discrete_map={'Subsidio': '#28a745', 'Impuesto': '#dc3545'},
                    title='Componente Fiscal por Sector (T)'
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(
                    variables_df[variables_df['country_code'] == selected_country][['sector', 'taxes_subsidies']],
                    use_container_width=True
                )
        
        # Mostrar convención de signos
        with st.expander("ℹ️ Ver convención de signos"):
            st.markdown(DOCS["tax_convention"])
    
    # ==========================================================================
    # TAB 3: MATRICES I-O
    # ==========================================================================
    with tab3:
        st.header("🔢 Matrices Input-Output")
        
        matrix_choice = st.selectbox(
            "Seleccionar matriz:",
            ["Z - Consumo Intermedio", "A - Coeficientes Técnicos", "L - Leontief"]
        )
        
        # Mostrar documentación según la matriz seleccionada
        if matrix_choice == "Z - Consumo Intermedio":
            matrix_data = analyzer.Z
            with st.expander("ℹ️ ¿Qué es la Matriz Z?", expanded=True):
                st.markdown(DOCS["matrix_z"])
        elif matrix_choice == "A - Coeficientes Técnicos":
            matrix_data = analyzer.A
            with st.expander("ℹ️ ¿Qué es la Matriz A?", expanded=True):
                st.markdown(DOCS["matrix_a"])
        else:
            matrix_data = analyzer.L
            with st.expander("ℹ️ ¿Qué es la Matriz L?", expanded=True):
                st.markdown(DOCS["matrix_l"])
        
        if PLOTLY_AVAILABLE:
            fig = px.imshow(
                matrix_data,
                x=analyzer.sectors,
                y=analyzer.sectors,
                color_continuous_scale='Blues',
                title=matrix_choice,
                labels={'x': 'Sector (destino/demandante)', 'y': 'Sector (origen/proveedor)', 'color': 'Valor'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Ver matriz en formato tabla"):
            df_matrix = pd.DataFrame(matrix_data, index=analyzer.sectors, columns=analyzer.sectors)
            st.dataframe(df_matrix.round(4), use_container_width=True)
    
    # ==========================================================================
    # TAB 4: MULTIPLICADORES
    # ==========================================================================
    with tab4:
        st.header("📊 Multiplicadores y Clasificación Sectorial")
        
        # Mostrar documentación
        with st.expander("ℹ️ ¿Cómo se calculan e interpretan?", expanded=False):
            st.markdown(DOCS["multipliers"])
        
        multipliers_df = analyzer.compute_multipliers()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tabla de Multiplicadores")
            st.dataframe(multipliers_df.round(4), use_container_width=True)
        
        with col2:
            max_mult_idx = multipliers_df['type_I_multiplier'].idxmax()
            min_mult_idx = multipliers_df['type_I_multiplier'].idxmin()
            
            st.metric(
                "Mayor Multiplicador",
                f"{multipliers_df.loc[max_mult_idx, 'sector']}",
                f"{multipliers_df.loc[max_mult_idx, 'type_I_multiplier']:.4f}"
            )
            st.metric(
                "Menor Multiplicador",
                f"{multipliers_df.loc[min_mult_idx, 'sector']}",
                f"{multipliers_df.loc[min_mult_idx, 'type_I_multiplier']:.4f}"
            )
        
        st.subheader("Diagrama de Clasificación Sectorial")
        
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
                title='Clasificación Sectorial por Linkages'
            )
            fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5,
                         annotation_text="FL promedio", annotation_position="right")
            fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5,
                         annotation_text="BL promedio", annotation_position="top")
            fig.update_traces(textposition='top center')
            fig.update_layout(
                height=500,
                xaxis_title="Backward Linkage (normalizado) - Importancia como DEMANDANTE",
                yaxis_title="Forward Linkage (normalizado) - Importancia como PROVEEDOR"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # TAB 5: ANÁLISIS HEF
    # ==========================================================================
    with tab5:
        st.header("💰 Extracción Hipotética Fiscal (HEF)")
        
        # Mostrar documentación
        with st.expander("ℹ️ ¿Qué es el método HEF y cómo funciona?", expanded=False):
            st.markdown(DOCS["hef_method"])
        
        st.markdown(f"**Parámetro actual:** Elasticidad precio = `{elasticity}`")
        
        hef_results = analyzer.fiscal_hypothetical_extraction(elasticity=elasticity)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Resultados HEF")
            display_hef = hef_results[['sector', 'tax_type', 'tax_original', 'tax_rate_pct', 'relative_impact_pct']].copy()
            st.dataframe(display_hef.round(4), use_container_width=True)
        
        with col2:
            max_impact_idx = hef_results['relative_impact_pct'].abs().idxmax()
            max_row = hef_results.loc[max_impact_idx]
            
            st.metric(
                "Sector con Mayor Impacto Sistémico",
                max_row['sector'],
                f"{max_row['relative_impact_pct']:+.4f}%"
            )
            
            st.info("""
            **Interpretación de signos:**
            - Eliminar **IMPUESTO** (T<0) → Costos ↓ → Producción **↑**
            - Eliminar **SUBSIDIO** (T>0) → Costos ↑ → Producción **↓**
            """)
        
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
                    title='Impacto de ELIMINAR el Componente Fiscal'
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(xaxis_title="Cambio en Producción Total (%)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    hef_results,
                    x='tax_original',
                    y='relative_impact_pct',
                    text='sector',
                    color='tax_type',
                    size=hef_results['tax_original'].abs(),
                    color_discrete_map={'Subsidio': '#28a745', 'Impuesto': '#dc3545'},
                    title='Relación: T Original vs Impacto HEF'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.update_traces(textposition='top center')
                fig.update_layout(
                    xaxis_title="Componente Fiscal Original (T) [millones USD]",
                    yaxis_title="Impacto de Eliminarlo (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Análisis detallado
        st.markdown("---")
        st.subheader("🔍 Análisis Detallado por Sector")
        
        selected_sector = st.selectbox("Seleccionar sector:", analyzer.sectors)
        sector_row = hef_results[hef_results['sector'] == selected_sector].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            tipo_emoji = "🟢" if sector_row['tax_type'] == 'Subsidio' else "🔴"
            st.metric(
                "Componente Fiscal", 
                f"{sector_row['tax_original']:+,.2f} M", 
                f"{tipo_emoji} {sector_row['tax_type']}"
            )
        with col2:
            st.metric("Tasa Fiscal", f"{sector_row['tax_rate_pct']:+.2f}%")
        with col3:
            st.metric("Impacto HEF", f"{sector_row['relative_impact_pct']:+.4f}%")
    
    # ==========================================================================
    # TAB 6: PROPAGACIÓN
    # ==========================================================================
    with tab6:
        st.header("🌊 Simulación de Propagación de Shocks")
        
        # Mostrar documentación
        with st.expander("ℹ️ ¿Cómo funciona la propagación de shocks?", expanded=False):
            st.markdown(DOCS["shock_propagation"])
        
        st.markdown(f"""
        **Parámetros actuales:**
        - Magnitud del shock: `{shock_magnitude*100:.0f}%` del output del sector
        - Rondas de propagación: `{shock_iterations}`
        """)
        
        shock_sector = st.selectbox("Sector origen del shock:", analyzer.sectors, key='shock_sector')
        shock_idx = analyzer.sector_to_idx[shock_sector]
        
        shock_result = analyzer.simulate_shock(shock_idx, shock_magnitude, shock_iterations)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Shock Inicial", f"{shock_magnitude * analyzer.X[shock_idx]:,.2f} M")
        with col2:
            st.metric("Efecto Total Acumulado", f"{shock_result['total_effect']:,.2f} M")
        with col3:
            initial = shock_magnitude * analyzer.X[shock_idx]
            if initial > 0:
                ratio = shock_result['total_effect'] / initial
                st.metric("Ratio Multiplicador", f"{ratio:.4f}x")
        
        if PLOTLY_AVAILABLE:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(shock_result['trajectory']))),
                    y=shock_result['trajectory'],
                    mode='lines+markers',
                    name='Efecto por ronda',
                    fill='tozeroy',
                    fillcolor='rgba(0,100,255,0.2)'
                ))
                fig.update_layout(
                    title='Propagación por Ronda',
                    xaxis_title='Ronda (k)',
                    yaxis_title='ε⁽ᵏ⁾ (millones USD)'
                )
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
                fig.update_layout(
                    title='Efecto Acumulado por Sector',
                    xaxis_title='Σₖ ε⁽ᵏ⁾ (millones USD)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Efectos de red
        st.markdown("---")
        st.subheader("🌐 Efectos de Red (Spillovers Fiscales)")
        
        with st.expander("ℹ️ ¿Qué son los efectos de red?", expanded=False):
            st.markdown(DOCS["network_effects"])
        
        network_effects = analyzer.compute_network_effects()
        
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                network_effects,
                x='sector',
                y=['own_tax_rate', 'network_effect'],
                barmode='group',
                title='Tasa Fiscal Propia vs Efecto de Red',
                labels={'value': 'Porcentaje (%)', 'sector': 'Sector', 'variable': 'Tipo'},
                color_discrete_map={'own_tax_rate': '#1f77b4', 'network_effect': '#ff7f0e'}
            )
            fig.update_layout(legend_title_text='')
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(network_effects.round(2), use_container_width=True)
    
    # ==========================================================================
    # TAB 7: VERIFICACIÓN
    # ==========================================================================
    with tab7:
        st.header("✅ Verificación de Cálculos")
        
        st.markdown("""
        Esta sección verifica que todos los cálculos cumplen las identidades fundamentales 
        del análisis Input-Output.
        """)
        
        checks = analyzer.verify_calculations()
        
        st.subheader("Identidades Fundamentales")
        
        for check_name, passed in checks.items():
            if passed:
                st.success(f"✅ {check_name}")
            else:
                st.error(f"❌ {check_name}")
        
        st.markdown("---")
        st.subheader("Resumen de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Balance de Producción:** `X = Ventas_Interm + Y`")
            balance_df = pd.DataFrame({
                'Sector': analyzer.sectors,
                'Σⱼ Z[i,j]': analyzer.Z.sum(axis=1),
                'Y[i]': analyzer.Y,
                'X[i]': analyzer.X,
                '✓': ['✅' if np.isclose(analyzer.Z.sum(axis=1)[i] + analyzer.Y[i], analyzer.X[i]) else '❌' 
                      for i in range(analyzer.n_sectors)]
            })
            st.dataframe(balance_df.round(2), use_container_width=True)
        
        with col2:
            st.markdown("**Balance Fiscal**")
            fiscal_data = {
                'Métrica': [
                    'Sectores subsidiados (T>0)',
                    'Sectores gravados (T<0)',
                    'Total subsidios',
                    'Total impuestos',
                    'Balance neto gobierno'
                ],
                'Valor': [
                    f"{(analyzer.T > 0).sum()} sectores",
                    f"{(analyzer.T < 0).sum()} sectores",
                    f"+{analyzer.T[analyzer.T > 0].sum():,.2f} M",
                    f"{analyzer.T[analyzer.T < 0].sum():,.2f} M",
                    f"{-analyzer.T.sum():+,.2f} M (recauda)" if analyzer.T.sum() < 0 else f"{-analyzer.T.sum():+,.2f} M (subsidia)"
                ]
            }
            st.dataframe(pd.DataFrame(fiscal_data), use_container_width=True)
        
        # Exportar resultados
        st.markdown("---")
        st.subheader("📥 Exportar Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            multipliers_csv = analyzer.compute_multipliers().to_csv(index=False)
            st.download_button(
                "📊 Descargar Multiplicadores",
                multipliers_csv,
                "multiplicadores.csv",
                "text/csv"
            )
        
        with col2:
            hef_export = analyzer.fiscal_hypothetical_extraction()
            hef_export = hef_export.drop(columns=['X_new'])
            hef_csv = hef_export.to_csv(index=False)
            st.download_button(
                "💰 Descargar Resultados HEF",
                hef_csv,
                "hef_results.csv",
                "text/csv"
            )
        
        with col3:
            network_csv = analyzer.compute_network_effects().to_csv(index=False)
            st.download_button(
                "🌐 Descargar Efectos de Red",
                network_csv,
                "network_effects.csv",
                "text/csv"
            )

# ============================================================================
# EJECUTAR APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    main()
