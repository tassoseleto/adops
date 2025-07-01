import streamlit as st
import pandas as pd
import os
import re
import plotly.express as px
from datetime import datetime
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Analisador de Desempenho por utm_source (GAM) - Versão 3.0")

# Configurações de cache para evitar reprocessamento desnecessário
@st.cache_data
def load_and_process_data(uploaded_files_info):
    """Carrega e processa os dados uma única vez"""
    all_data = []
    
    for file_info in uploaded_files_info:
        # Simula o carregamento - na prática, você precisará ajustar isso
        df = pd.read_csv(file_info['name'], thousands=',')
        df['source_file'] = file_info['basename']
        all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Padronização de colunas
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={
        'channel': 'utm_source',
        'date': 'data',
        'ad exchange active view viewable impression': 'ad exchange active view viewable impressions'
    }, inplace=True)
    
    # Conversão de data
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    
    # Processamento de métricas numéricas
    metricas_numericas = [
        'ad exchange impressions',
        'ad exchange clicks',
        'ad exchange revenue ($)',
        'ad exchange ad requests',
        'ad exchange active view measurable impressions',
        'ad exchange active view viewable impressions'
    ]
    for coluna in metricas_numericas:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors='coerce').fillna(0)
    # Cálculo das métricas derivadas
    df['CPC (US$)'] = (df['ad exchange revenue ($)'] / df['ad exchange clicks']).replace([np.inf, -np.inf], 0).fillna(0)
    df['eCPM (US$)'] = (df['ad exchange revenue ($)'] / df['ad exchange impressions']).replace([np.inf, -np.inf], 0).fillna(0) * 1000
    df['Match Rate (%)'] = (df['ad exchange impressions'] / df['ad exchange ad requests']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    df['CTR (%)'] = (df['ad exchange clicks'] / df['ad exchange impressions']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    df['Viewability (%)'] = (df['ad exchange active view viewable impressions'] / df['ad exchange active view measurable impressions']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    df['Ad Request eCPM (US$)'] = (df['ad exchange revenue ($)'] / df['ad exchange ad requests']).replace([np.inf, -np.inf], 0).fillna(0) * 1000
    # Remove linhas com dados inválidos críticos
    df = df.dropna(subset=['data', 'utm_source'])
    
    return df

def get_filtered_data(df_original, date_range, selected_sources, additional_filters=None):
    """Aplica filtros de forma consistente sem modificar o DataFrame original"""
    df_filtered = df_original.copy()
    
    # Filtro de data
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['data'] >= pd.to_datetime(date_range[0])) & 
            (df_filtered['data'] <= pd.to_datetime(date_range[1]))
        ]
    
    # Filtro de utm_source
    if selected_sources:
        df_filtered = df_filtered[df_filtered['utm_source'].isin(selected_sources)]
    
    # Filtros adicionais (para abas específicas)
    if additional_filters:
        for column, values in additional_filters.items():
            if column in df_filtered.columns and values:
                df_filtered = df_filtered[df_filtered[column].isin(values)]
    
    return df_filtered

def calculate_metrics_safely(df, group_columns, metrics_config):
    """Calcula métricas de forma segura, tratando valores nulos e infinitos"""
    if df.empty:
        return pd.DataFrame()
    
    # Agrupa e calcula métricas
    result = df.groupby(group_columns).agg(metrics_config).reset_index()
    
    # Trata valores infinitos e nulos
    for col in result.columns:
        if result[col].dtype in ['float64', 'int64']:
            result[col] = result[col].replace([np.inf, -np.inf], 0)
            result[col] = result[col].fillna(0)
    
    return result

def create_analysis_section(df, group_cols, metrics_config, title, key_prefix, metricas_disponiveis=None):
    """Cria uma seção de análise com métricas e gráficos"""
    # Adicionar data aos grupos se estiver na aba de Ad Unit
    if key_prefix == 'adunit' and 'data' not in group_cols:
        group_cols = ['data'] + group_cols

    # Seletor de métrica para gráfico e crescimento
    if metricas_disponiveis is None:
        if metrics_config is not None:
            metricas_disponiveis = list(metrics_config.keys())
        else:
            # Inferir métricas numéricas do DataFrame
            metricas_disponiveis = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    default_metric = 'ad exchange ad requests' if 'ad exchange ad requests' in metricas_disponiveis else (metricas_disponiveis[0] if metricas_disponiveis else None)
    selected_metric = st.selectbox(
        f"Selecione a métrica para o gráfico e análise de crescimento:",
        options=metricas_disponiveis,
        index=metricas_disponiveis.index(default_metric) if default_metric in metricas_disponiveis else 0,
        key=f"metric_growth_{key_prefix}"
    )

    # --- NOVO BLOCO: Agrupamento sempre com soma das bases e cálculo das derivadas ---
    base_cols = [
        'ad exchange impressions', 'ad exchange clicks', 'ad exchange revenue ($)',
        'ad exchange ad requests', 'ad exchange active view measurable impressions',
        'ad exchange active view viewable impressions'
    ]
    # Verifica se todas as colunas base existem no DataFrame
    for col in base_cols:
        if col not in df.columns:
            df[col] = 0
    # Agrupa pelas dimensões selecionadas
    grouped = df.groupby(group_cols)[base_cols].sum().reset_index()
    # Calcula as métricas derivadas
    grouped = calcular_metricas_derivadas(grouped, group_cols)
    # Seleciona as métricas para exibir (base + derivadas)
    all_metrics = base_cols + ['CTR (%)', 'CPC (US$)', 'eCPM (US$)', 'Ad Request eCPM (US$)', 'Match Rate (%)']
    # Filtra para mostrar apenas as métricas selecionadas pelo usuário (se houver)
    if metrics_config:
        selected_metrics = list(metrics_config.keys())
        display_cols = group_cols + [m for m in all_metrics if m in selected_metrics or m in grouped.columns]
    else:
        display_cols = group_cols + all_metrics
    metrics = grouped[display_cols]
    # --- FIM NOVO BLOCO ---
    
    if not metrics.empty:
        st.write(f"**{title}:**")
        
        # --- NOVO BLOCO: Seletor de ordenação ---
        order_options = [col for col in metrics.columns if col not in group_cols and pd.api.types.is_numeric_dtype(metrics[col])]
        default_order = 'ad exchange ad requests' if 'ad exchange ad requests' in order_options else (order_options[0] if order_options else None)
        order_by = st.selectbox(
            'Ordenar tabela por:',
            options=order_options,
            index=order_options.index(default_order) if default_order in order_options else 0,
            key=f"order_by_{key_prefix}"
        ) if order_options else None
        if order_by:
            metrics = metrics.sort_values(order_by, ascending=False)
            if key_prefix == 'url':
                metrics = metrics.head(10)
        # --- FIM NOVO BLOCO ---
        
        # Formatação da tabela
        format_dict = {}
        for col in metrics.columns:
            if col in group_cols:
                if col == 'data':
                    format_dict[col] = lambda x: x.strftime('%d/%m/%Y') if pd.notnull(x) else ''
                continue  # Pula outras colunas de dimensão
            if pd.api.types.is_numeric_dtype(metrics[col]):
                if any(x in col.lower() for x in ['ad requests', 'impressions', 'clicks']):
                    format_dict[col] = '{:,.0f}'
                elif any(x in col.lower() for x in ['rate', 'ctr', 'viewable', '%']):
                    format_dict[col] = '{:.2f}%'
                elif any(x in col.lower() for x in ['ecpm', 'cpc', '$', 'revenue']):
                    format_dict[col] = 'US$ {:.2f}'
                else:
                    format_dict[col] = '{:.2f}'
        
        # Definir colunas de dimensão para fixar à esquerda
        dimensoes_fixas = [col for col in group_cols if col in metrics.columns]
        try:
            import streamlit as stlib
            column_config = {col: st.column_config.Column(label=col, pinned="left") for col in dimensoes_fixas}
        except Exception:
            column_config = None
        
        # Aplicar formatação apenas nas colunas numéricas
        styled_metrics = metrics.style.format(format_dict)
        
        # Aplicar gradiente apenas nas colunas numéricas
        numeric_cols = [col for col in metrics.columns if pd.api.types.is_numeric_dtype(metrics[col])]
        if numeric_cols:
            styled_metrics = styled_metrics.background_gradient(
                cmap="RdYlGn",
                subset=numeric_cols
            )
        
        st.dataframe(
            styled_metrics,
            use_container_width=True,
            column_config=column_config if column_config else None
        )
        
        # Gráfico de evolução
        if selected_metric in df.columns:
            # Determinar as colunas para agrupamento baseado no tipo de análise
            if key_prefix == 'adtype' and 'ad type' in df.columns:
                group_by_cols = ['data', 'ad type']
                color_col = 'ad type'
                df_graph = df.copy()
            elif key_prefix == 'advertiser' and 'advertiser (classified)' in df.columns:
                group_by_cols = ['data', 'advertiser (classified)']
                color_col = 'advertiser (classified)'
                # Filtrar para os top 5 advertisers por total da métrica selecionada
                top_advertisers = df.groupby('advertiser (classified)')[selected_metric].sum().nlargest(5).index
                df_graph = df[df['advertiser (classified)'].isin(top_advertisers)].copy()
            elif key_prefix == 'url' and 'url' in df.columns:
                group_by_cols = ['data', 'url']
                color_col = 'url'
                # Filtrar para as TOP 10 URLs exibidas na tabela
                top_urls = metrics['url'].unique() if 'url' in metrics.columns else []
                df_graph = df[df['url'].isin(top_urls)].copy()
            else:
                group_by_cols = ['data'] + [col for col in group_cols if col != 'data']
                color_col = group_cols[1] if len(group_cols) > 1 else group_cols[0]
                df_graph = df.copy()

            # Métricas derivadas que precisam ser recalculadas a partir dos totais
            metricas_derivadas = ['CPC (US$)', 'eCPM (US$)', 'CTR (%)', 'Ad Request eCPM (US$)', 'Match Rate (%)']
            base_cols = [
                'ad exchange impressions', 'ad exchange clicks', 'ad exchange revenue ($)',
                'ad exchange ad requests', 'ad exchange active view measurable impressions',
                'ad exchange active view viewable impressions'
            ]
            if selected_metric in metricas_derivadas:
                # Agrupar e somar as bases, depois calcular a métrica derivada
                grouped_graph = df_graph.groupby(group_by_cols)[base_cols].sum().reset_index()
                grouped_graph = calcular_metricas_derivadas(grouped_graph, group_by_cols)
                trend = grouped_graph[group_by_cols + [selected_metric]]
            else:
                # Métricas base: pode usar soma ou média conforme apropriado
                metricas_soma = [
                    'ad exchange ad requests',
                    'ad exchange revenue ($)',
                    'ad exchange impressions',
                    'ad exchange clicks',
                    'ad exchange active view measurable impressions',
                    'ad exchange active view viewable impressions'
                ]
                if selected_metric.lower() in metricas_soma:
                    agg_func = 'sum'
                else:
                    agg_func = 'mean'
                trend = calculate_metrics_safely(
                    df_graph,
                    group_by_cols,
                    {selected_metric: agg_func}
                )

            if not trend.empty:
                # Configurar cores para melhor visualização
                colors = px.colors.qualitative.Set3
                # Criar o gráfico
                fig = px.line(
                    trend,
                    x='data',
                    y=selected_metric,
                    color=color_col,
                    markers=True,
                    title=f'Evolução de {selected_metric} - {title}',
                    color_discrete_sequence=colors
                )
                
                # Melhorar o layout
                fig.update_layout(
                    xaxis_title='Data',
                    yaxis_title=selected_metric,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    margin=dict(t=50, l=50, r=50, b=50)
                )
                
                # Formatar eixos
                fig.update_xaxes(
                    tickangle=45,
                    tickformat='%d/%m/%Y',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                )
                
                fig.update_yaxes(
                    tickformat=',.2f',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                )
                
                # Melhorar tooltips
                fig.update_traces(
                    hovertemplate='<b>%{x|%d/%m/%Y}</b><br>' +
                                f'{selected_metric}: '+'%{y:,.2f}<br>' +
                                '<extra></extra>'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análise de Crescimento SEMPRE aparece
                st.subheader("📈 Análise de Crescimento")
                growth_data = []
                for group in trend[color_col].unique():
                    group_data = trend[trend[color_col] == group].copy()
                    group_data = group_data.sort_values('data')
                    first_value = group_data[selected_metric].iloc[0]
                    last_value = group_data[selected_metric].iloc[-1]
                    growth_pct = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                    growth_info = {
                        'Grupo': group,
                        'Valor Inicial': first_value,
                        'Valor Final': last_value,
                        'Crescimento Absoluto': last_value - first_value,
                        'Crescimento %': growth_pct  # float, não string
                    }
                    growth_data.append(growth_info)
                if growth_data:
                    growth_df = pd.DataFrame(growth_data)
                    growth_df = growth_df.sort_values('Crescimento %', ascending=False)
                    format_dict = {
                        'Valor Inicial': '{:,.2f}',
                        'Valor Final': '{:,.2f}',
                        'Crescimento Absoluto': '{:,.2f}',
                        'Crescimento %': '{:+.2f}%'
                    }
                    growth_styled = growth_df.style.format(format_dict).background_gradient(
                        subset=['Crescimento %'],
                        cmap='RdYlGn'
                    )
                    st.dataframe(growth_styled, use_container_width=True)
        
        # Export
        if st.button(f"\U0001F4E4 Exportar dados de {title}", key=f"export_{key_prefix}"):
            export_cols = group_cols.copy()
            if metrics_config is not None:
                export_cols.extend(metrics_config.keys())
            else:
                # Inferir métricas numéricas do DataFrame agrupado
                export_cols.extend([col for col in metrics.columns if col not in group_cols])
            csv_data = metrics[export_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                f"\U0001F4E5 Baixar CSV - {title}",
                csv_data,
                f"analise_{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                key=f"download_{key_prefix}"
            )
    else:
        st.warning("\u26A0\uFE0F Nenhum dado encontrado com a configuração atual.")

def create_url_filter(df, key_prefix):
    """Cria um filtro de URLs"""
    if 'url' in df.columns:
        st.write("**Filtro Adicional:**")
        urls_disponiveis = sorted(df['url'].dropna().unique())
        return st.multiselect(
            "Filtrar por URLs específicas (opcional):",
            options=urls_disponiveis,
            key=f"url_filter_{key_prefix}"
        )
    return []

def create_dimension_selector(df, base_dims, key_prefix):
    """Cria um seletor de dimensões"""
    dimensoes_disponiveis = base_dims.copy()
    if 'url' in df.columns:
        dimensoes_disponiveis.append('url')
    
    return st.multiselect(
        "Dimensões para agrupamento:",
        options=dimensoes_disponiveis,
        default=base_dims,
        key=f"dimensoes_{key_prefix}"
    )

def create_source_filter(df, key_prefix):
    """Cria um filtro de sources"""
    sources_disponiveis = sorted(df['utm_source'].dropna().unique())
    return st.multiselect(
        "Filtrar por Sources:",
        options=sources_disponiveis,
        default=sources_disponiveis[:5] if len(sources_disponiveis) <= 5 else sources_disponiveis[:3],
        key=f"source_filter_{key_prefix}"
    )

# Função para calcular métricas derivadas a partir dos totais agrupados
def calcular_metricas_derivadas(df, group_cols):
    # Garante que todas as colunas base existem
    for col in [
        'ad exchange impressions', 'ad exchange clicks', 'ad exchange revenue ($)',
        'ad exchange ad requests', 'ad exchange active view measurable impressions',
        'ad exchange active view viewable impressions'
    ]:
        if col not in df.columns:
            df[col] = 0
    grouped = df.groupby(group_cols).agg({
        'ad exchange impressions': 'sum',
        'ad exchange clicks': 'sum',
        'ad exchange revenue ($)': 'sum',
        'ad exchange ad requests': 'sum',
        'ad exchange active view measurable impressions': 'sum',
        'ad exchange active view viewable impressions': 'sum'
    }).reset_index()
    # Converter colunas base para numérico após o agrupamento
    for col in [
        'ad exchange impressions', 'ad exchange clicks', 'ad exchange revenue ($)',
        'ad exchange ad requests', 'ad exchange active view measurable impressions',
        'ad exchange active view viewable impressions'
    ]:
        if col in grouped.columns:
            grouped[col] = pd.to_numeric(grouped[col], errors='coerce').fillna(0)
    # Métricas derivadas
    grouped['CTR (%)'] = (grouped['ad exchange clicks'] / grouped['ad exchange impressions']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    grouped['CPC (US$)'] = (grouped['ad exchange revenue ($)'] / grouped['ad exchange clicks']).replace([np.inf, -np.inf], 0).fillna(0)
    grouped['eCPM (US$)'] = (grouped['ad exchange revenue ($)'] / grouped['ad exchange impressions']).replace([np.inf, -np.inf], 0).fillna(0) * 1000
    grouped['Ad Request eCPM (US$)'] = (grouped['ad exchange revenue ($)'] / grouped['ad exchange ad requests']).replace([np.inf, -np.inf], 0).fillna(0) * 1000
    grouped['Match Rate (%)'] = (grouped['ad exchange impressions'] / grouped['ad exchange ad requests']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    return grouped

# Bloco universal de formatação para todas as tabelas
universal_format_dict = {}
def build_format_dict(df):
    format_dict = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_lower = col.lower()
            if any(x in col_lower for x in ['ecpm', 'cpc', '$', 'revenue']):
                format_dict[col] = 'US$ {:.2f}'
            elif any(x in col_lower for x in ['rate', 'ctr', 'viewable', '%']):
                format_dict[col] = '{:.2f}%'
            elif any(x in col_lower for x in ['impressions', 'clicks', 'requests']):
                format_dict[col] = '{:,.0f}'
            else:
                format_dict[col] = '{:.2f}'
    return format_dict

# Interface de upload
uploaded_files = st.file_uploader(
    "📅 Faça upload dos CSVs exportados do GAM", 
    accept_multiple_files=True, 
    type='csv'
)

if uploaded_files:
    # Preparar informações dos arquivos para cache
    files_info = [
        {'name': file, 'basename': os.path.basename(file.name)} 
        for file in uploaded_files
    ]
    
    # Carregar dados (com cache)
    try:
        df_original = load_and_process_data(tuple(str(f) for f in files_info))
    except Exception as e:
        # Fallback sem cache para desenvolvimento
        all_data = []
        for file in uploaded_files:
            df = pd.read_csv(file, thousands=',')
            df['source_file'] = os.path.basename(file.name)
            all_data.append(df)
        
        df_original = pd.concat(all_data, ignore_index=True)
        
        # Padronização
        df_original.columns = [c.strip().lower() for c in df_original.columns]
        df_original.rename(columns={'channel': 'utm_source', 'date': 'data', 'ad exchange active view viewable impression': 'ad exchange active view viewable impressions'}, inplace=True)
        df_original['data'] = pd.to_datetime(df_original['data'], errors='coerce')
        
        # Processamento numérico
        metricas_numericas = [
            'ad exchange impressions',
            'ad exchange clicks',
            'ad exchange revenue ($)',
            'ad exchange ad requests',
            'ad exchange active view measurable impressions',
            'ad exchange active view viewable impressions'
        ]
        for coluna in metricas_numericas:
            if coluna in df_original.columns:
                df_original[coluna] = pd.to_numeric(df_original[coluna], errors='coerce').fillna(0)
        # Cálculo das métricas derivadas
        df_original['CPC (US$)'] = (df_original['ad exchange revenue ($)'] / df_original['ad exchange clicks']).replace([np.inf, -np.inf], 0).fillna(0)
        df_original['eCPM (US$)'] = (df_original['ad exchange revenue ($)'] / df_original['ad exchange impressions']).replace([np.inf, -np.inf], 0).fillna(0) * 1000
        df_original['Match Rate (%)'] = (df_original['ad exchange impressions'] / df_original['ad exchange ad requests']).replace([np.inf, -np.inf], 0).fillna(0) * 100
        df_original['CTR (%)'] = (df_original['ad exchange clicks'] / df_original['ad exchange impressions']).replace([np.inf, -np.inf], 0).fillna(0) * 100
        df_original['Viewability (%)'] = (df_original['ad exchange active view viewable impressions'] / df_original['ad exchange active view measurable impressions']).replace([np.inf, -np.inf], 0).fillna(0) * 100
        df_original['Ad Request eCPM (US$)'] = (df_original['ad exchange revenue ($)'] / df_original['ad exchange ad requests']).replace([np.inf, -np.inf], 0).fillna(0) * 1000
        df_original = df_original.dropna(subset=['data', 'utm_source'])
    
    # Validação de colunas obrigatórias
    required_columns = ['utm_source', 'data']
    missing_columns = [col for col in required_columns if col not in df_original.columns]
    
    if missing_columns:
        st.error(f"❌ Colunas obrigatórias não encontradas: {missing_columns}")
        st.stop()
    
    # Configuração de métricas
    metricas_config = {
        'ad exchange ad requests': 'sum',
        'ad exchange match rate': 'mean',
        'ad exchange ad request ecpm ($)': 'mean',
        'ad exchange cpc ($)': 'mean',
        'ad exchange ctr': 'mean',
        'ad exchange active view % viewable impressions': 'mean'
    }
    
    # Identificar métricas disponíveis
    metricas_disponiveis = [m for m in metricas_config.keys() if m in df_original.columns]
    
    if not metricas_disponiveis:
        st.error("❌ Nenhuma métrica esperada foi encontrada nos dados.")
        st.stop()
    
    # Sidebar com informações dos dados
    with st.sidebar:
        st.header("ℹ️ Informações dos Dados")
        st.write(f"**Total de registros:** {len(df_original):,}")
        st.write(f"**Período:** {df_original['data'].min().strftime('%d/%m/%Y')} a {df_original['data'].max().strftime('%d/%m/%Y')}")
        st.write(f"**UTM Sources:** {df_original['utm_source'].nunique()}")
        if 'url' in df_original.columns:
            st.write(f"**URLs únicas:** {df_original['url'].nunique()}")
        st.write(f"**Métricas disponíveis:** {len(metricas_disponiveis)}")
    
    # Filtros principais
    st.header("🔍 Filtros Principais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Filtro de data
        data_min = df_original['data'].min().date()
        data_max = df_original['data'].max().date()
        date_range = st.date_input(
            "🗓 Período de análise:",
            value=[data_min, data_max],
            min_value=data_min,
            max_value=data_max,
            format="DD/MM/YYYY"
        )
    
    with col2:
        # Filtro de utm_source principal (simples)
        todas_sources = sorted(df_original['utm_source'].dropna().unique())
        selected_sources = st.multiselect(
            "🌐 Canais (utm_source):",
            options=todas_sources,
            default=todas_sources[:5] if len(todas_sources) <= 5 else todas_sources[:3],
            key="source_filter_main"
        )
    
    if not selected_sources:
        st.warning("⚠️ Selecione ao menos um canal para continuar a análise.")
        st.stop()
    
    # Aplicar filtros principais
    df_filtered = get_filtered_data(df_original, date_range, selected_sources)
    
    if df_filtered.empty:
        st.warning("⚠️ Nenhum dado encontrado com os filtros aplicados.")
        st.stop()
    
    # Abas de análise
    st.header("📊 Análises Detalhadas")

    tab_geral, tab_source, tab_url, tab_adunit, tab_adtype, tab_advertiser = st.tabs([
        "📊 Visão Geral",
        "🌐 Source",
        "🔗 URL",
        "📦 Ad Unit",
        "📱 Ad Type",
        "🏢 Advertiser"
    ])

    with tab_geral:
        st.subheader("📊 Visão Geral do Report")
        # 1. Período do report
        periodo = f"{df_filtered['data'].min().strftime('%d/%m/%Y')} a {df_filtered['data'].max().strftime('%d/%m/%Y')}"
        st.markdown(f"**Período do Report:** {periodo}")

        # 2. KPIs principais
        receita_total = df_filtered['ad exchange revenue ($)'].sum() if 'ad exchange revenue ($)' in df_filtered.columns else 0
        impressoes_total = df_filtered['ad exchange impressions'].sum() if 'ad exchange impressions' in df_filtered.columns else 0
        cliques_total = df_filtered['ad exchange clicks'].sum() if 'ad exchange clicks' in df_filtered.columns else 0
        ad_requests_total = df_filtered['ad exchange ad requests'].sum() if 'ad exchange ad requests' in df_filtered.columns else 0
        ecpm_medio = df_filtered['eCPM (US$)'].mean() if 'eCPM (US$)' in df_filtered.columns else 0
        cpc_medio = df_filtered['CPC (US$)'].mean() if 'CPC (US$)' in df_filtered.columns else 0
        ctr_medio = df_filtered['CTR (%)'].mean() if 'CTR (%)' in df_filtered.columns else 0
        matchrate_medio = df_filtered['Match Rate (%)'].mean() if 'Match Rate (%)' in df_filtered.columns else 0
        viewability_medio = df_filtered['Viewability (%)'].mean() if 'Viewability (%)' in df_filtered.columns else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Receita Total", f"US$ {receita_total:,.2f}")
        col2.metric("Impressões", f"{impressoes_total:,.0f}")
        col3.metric("Cliques", f"{cliques_total:,.0f}")
        col4.metric("Ad Requests", f"{ad_requests_total:,.0f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("eCPM Médio", f"US$ {ecpm_medio:,.2f}")
        col6.metric("CPC Médio", f"US$ {cpc_medio:,.2f}")
        col7.metric("CTR Médio", f"{ctr_medio:,.2f}%")
        col8.metric("Match Rate Médio", f"{matchrate_medio:,.2f}%")
        st.metric("Viewability Média", f"{viewability_medio:,.2f}%")

        # 3. Gráfico Receita vs. eCPM
        if 'data' in df_filtered.columns and 'ad exchange revenue ($)' in df_filtered.columns and 'eCPM (US$)' in df_filtered.columns:
            st.subheader("Gráfico Receita vs. eCPM (por dia)")
            df_trend = df_filtered.groupby('data').agg({
                'ad exchange revenue ($)': 'sum',
                'eCPM (US$)': 'mean'
            }).reset_index()
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_trend['data'], y=df_trend['ad exchange revenue ($)'], name='Receita', yaxis='y1'))
            fig.add_trace(go.Line(x=df_trend['data'], y=df_trend['eCPM (US$)'], name='eCPM', yaxis='y2', marker_color='orange'))
            fig.update_layout(
                xaxis_title='Data',
                yaxis=dict(title='Receita (US$)', side='left', showgrid=False),
                yaxis2=dict(title='eCPM (US$)', overlaying='y', side='right', showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Sugestão: Top 5 UTM Sources por Receita
        if 'utm_source' in df_filtered.columns and 'ad exchange revenue ($)' in df_filtered.columns:
            st.subheader("Top 5 UTM Sources por Receita")
            top_sources = df_filtered.groupby('utm_source')['ad exchange revenue ($)'].sum().sort_values(ascending=False).head(5)
            st.dataframe(top_sources.reset_index().rename(columns={'ad exchange revenue ($)': 'Receita (US$)'}), use_container_width=True)

        # Sugestão: Top 5 URLs por Receita
        if 'url' in df_filtered.columns and 'ad exchange revenue ($)' in df_filtered.columns:
            st.subheader("Top 5 URLs por Receita")
            top_urls = df_filtered.groupby('url')['ad exchange revenue ($)'].sum().sort_values(ascending=False).head(5)
            st.dataframe(top_urls.reset_index().rename(columns={'ad exchange revenue ($)': 'Receita (US$)'}), use_container_width=True)
            # Gráfico de linhas de receita ao longo do tempo para as Top 5 URLs
            st.subheader("Evolução da Receita das Top 5 URLs")
            top_url_list = top_urls.index.tolist()
            df_top_urls = df_filtered[df_filtered['url'].isin(top_url_list)]
            if not df_top_urls.empty:
                df_trend_url = df_top_urls.groupby(['data', 'url'])['ad exchange revenue ($)'].sum().reset_index()
                fig_url = px.line(
                    df_trend_url,
                    x='data',
                    y='ad exchange revenue ($)',
                    color='url',
                    markers=True,
                    title='Receita Diária das Top 5 URLs',
                    labels={'ad exchange revenue ($)': 'Receita (US$)', 'data': 'Data', 'url': 'URL'}
                )
                fig_url.update_layout(
                    xaxis_title='Data',
                    yaxis_title='Receita (US$)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_url, use_container_width=True)

        # Espaço para mais sugestões e gráficos
        st.info("Adicione mais gráficos ou análises executivas conforme desejar!")

    with tab_source:
        st.subheader("📋 Consolidado por UTM Source")
        # (aba Visão Geral antiga, só muda o nome)
        metrics_to_agg = {m: metricas_config[m] for m in metricas_disponiveis}
        consolidado = calculate_metrics_safely(
            df_filtered, 
            ['utm_source'], 
            metrics_to_agg
        )
        if not consolidado.empty:
            total_row = {}
            for col in metricas_disponiveis:
                if metricas_config[col] == 'sum':
                    total_row[col] = consolidado[col].sum()
                else:
                    if 'ad exchange ad requests' in consolidado.columns:
                        weights = consolidado['ad exchange ad requests']
                        total_row[col] = np.average(consolidado[col], weights=weights)
                    else:
                        total_row[col] = consolidado[col].mean()
            total_row['utm_source'] = 'TOTAL GERAL'
            total_df = pd.DataFrame([total_row])
            consolidado = pd.concat([consolidado, total_df], ignore_index=True)
        display_columns = {
            'ad exchange ad requests': 'Ad Requests',
            'ad exchange match rate': 'Match Rate (%)',
            'ad exchange ad request ecpm ($)': 'eCPM (US$)',
            'ad exchange cpc ($)': 'CPC (US$)',
            'ad exchange ctr': 'CTR (%)',
            'ad exchange active view % viewable impressions': 'Viewability (%)'
        }
        consolidado_display = consolidado.rename(columns=display_columns)
        format_dict = build_format_dict(consolidado_display)
        st.dataframe(consolidado_display.style.format(format_dict), use_container_width=True)
        if 'ad exchange ad requests' in df_filtered.columns:
            st.subheader("📈 Evolução Temporal - Ad Requests")
            trend_data = calculate_metrics_safely(
                df_filtered,
                ['data', 'utm_source'],
                {'ad exchange ad requests': 'sum'}
            )
            if not trend_data.empty:
                fig = px.line(
                    trend_data,
                    x='data',
                    y='ad exchange ad requests',
                    color='utm_source',
                    markers=True,
                    title='Evolução Diária de Ad Requests por UTM Source'
                )
                fig.update_layout(
                    xaxis_title='Data',
                    yaxis_title='Ad Requests',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab_url:
        if 'url' not in df_filtered.columns:
            st.warning("⚠️ Coluna 'URL' não encontrada nos dados.")
        else:
            key_prefix = 'url'
            selected_urls = create_url_filter(df_filtered, key_prefix=key_prefix)
            selected_sources = create_source_filter(df_filtered, key_prefix=key_prefix)
            base_dims = ['ad unit', 'utm_source'] if 'ad unit' in df_filtered.columns else ['utm_source']
            group_cols = create_dimension_selector(df_filtered, base_dims=base_dims, key_prefix=key_prefix)
            if 'url' not in group_cols:
                group_cols = ['url'] + group_cols
            additional_filters = {}
            if selected_urls:
                additional_filters['url'] = selected_urls
            if selected_sources:
                additional_filters['utm_source'] = selected_sources
            df_tab = get_filtered_data(df_filtered, date_range, selected_sources, additional_filters)
            create_analysis_section(
                df_tab,
                group_cols=group_cols,
                metrics_config=None,
                title='Métricas por URL',
                key_prefix=key_prefix
            )

    with tab_adunit:
        if 'ad unit' not in df_filtered.columns:
            st.warning("⚠️ Coluna 'Ad Unit' não encontrada nos dados.")
        else:
            key_prefix = 'adunit'
            selected_urls = create_url_filter(df_filtered, key_prefix=key_prefix)
            selected_sources = create_source_filter(df_filtered, key_prefix=key_prefix)
            adunits_disponiveis = sorted(df_filtered['ad unit'].dropna().unique())
            selected_adunits = st.multiselect(
                "Filtrar por Ad Units:",
                options=adunits_disponiveis,
                key=f"adunit_filter_{key_prefix}"
            )
            base_dims = ['ad unit', 'utm_source']
            group_cols = create_dimension_selector(df_filtered, base_dims=base_dims, key_prefix=key_prefix)
            additional_filters = {}
            if selected_urls:
                additional_filters['url'] = selected_urls
            if selected_adunits:
                additional_filters['ad unit'] = selected_adunits
            if selected_sources:
                additional_filters['utm_source'] = selected_sources
            df_tab = get_filtered_data(df_filtered, date_range, selected_sources, additional_filters)
            create_analysis_section(
                df_tab,
                group_cols=group_cols,
                metrics_config=None,
                title='Métricas por Ad Unit',
                key_prefix=key_prefix
            )

    with tab_adtype:
        if 'ad type' not in df_filtered.columns:
            st.warning("⚠️ Coluna 'Ad Type' não encontrada nos dados.")
        else:
            key_prefix = 'adtype'
            selected_sources = create_source_filter(df_filtered, key_prefix=key_prefix)
            base_dims = ['ad type', 'utm_source']
            group_cols = create_dimension_selector(df_filtered, base_dims=base_dims, key_prefix=key_prefix)
            additional_filters = {}
            if selected_sources:
                additional_filters['utm_source'] = selected_sources
            df_tab = get_filtered_data(df_filtered, date_range, selected_sources, additional_filters)
            create_analysis_section(
                df_tab,
                group_cols=group_cols,
                metrics_config=None,
                title='Métricas por Ad Type',
                key_prefix=key_prefix
            )

    with tab_advertiser:
        if 'advertiser (classified)' not in df_filtered.columns:
            st.warning("⚠️ Coluna 'Advertiser (classified)' não encontrada nos dados.")
        else:
            key_prefix = 'advertiser'
            selected_sources = create_source_filter(df_filtered, key_prefix=key_prefix)
            base_dims = ['advertiser (classified)', 'utm_source']
            group_cols = create_dimension_selector(df_filtered, base_dims=base_dims, key_prefix=key_prefix)
            additional_filters = {}
            if selected_sources:
                additional_filters['utm_source'] = selected_sources
            df_tab = get_filtered_data(df_filtered, date_range, selected_sources, additional_filters)
            create_analysis_section(
                df_tab,
                group_cols=group_cols,
                metrics_config=None,
                title='Métricas por Advertiser',
                key_prefix=key_prefix
            )

    # Export geral
    st.header("⬇️ Exportar Dados")
    
    export_columns = ['data', 'utm_source']
    if 'url' in df_filtered.columns:
        export_columns.append('url')
    if 'ad unit' in df_filtered.columns:
        export_columns.append('ad unit')
    
    export_columns.extend(metricas_disponiveis)
    
    export_df = df_filtered[export_columns].copy()
    
    # Informações do export
    st.write(f"**Registros para export:** {len(export_df):,}")
    st.write(f"**Período:** {export_df['data'].min().strftime('%d/%m/%Y')} a {export_df['data'].max().strftime('%d/%m/%Y')}")
    
    csv_export = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📤 Baixar dados filtrados (CSV)",
        csv_export,
        f"analise_gam_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True
    )

else:
    st.info("👆 Faça upload dos arquivos CSV para começar a análise.")
    
    # Informações sobre o formato esperado
    with st.expander("ℹ️ Formato de dados esperado"):
        st.write("""
        **Colunas obrigatórias:**
        - `Channel` ou `utm_source`: Canal de origem
        - `Date` ou `data`: Data do registro
        
        **Colunas opcionais:**
        - `URL`: URL analisada
        - `Ad Unit`: Unidade de anúncio
        - `Ad Type`: Tipo de anúncio
        - `Advertiser (classified)`: Anunciante classificado
        
        **Métricas esperadas:**
        - `Ad Exchange Ad Requests`: Solicitações de anúncios
        - `Ad Exchange Match Rate`: Taxa de correspondência
        - `Ad Exchange Ad Request ECPM ($)`: eCPM por solicitação
        - `Ad Exchange CPC ($)`: Custo por clique
        - `Ad Exchange CTR`: Taxa de cliques
        - `Ad Exchange Active View % Viewable Impressions`: Porcentagem de visualizações
        """)
