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
        df = pd.read_csv(file_info['name'])
        df['source_file'] = file_info['basename']
        all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Padronização de colunas
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={
        'channel': 'utm_source',
        'date': 'data'
    }, inplace=True)
    
    # Conversão de data
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    
    # Processamento de métricas numéricas
    metricas_numericas = [
        'ad exchange ad requests',
        'ad exchange match rate',
        'ad exchange ad request ecpm ($)',
        'ad exchange cpc ($)',
        'ad exchange ctr',
        'ad exchange active view % viewable impressions'
    ]
    
    for coluna in metricas_numericas:
        if coluna in df.columns:
            # Limpa e converte para numérico
            df[coluna] = df[coluna].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
            df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
    
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
        metricas_disponiveis = list(metrics_config.keys())
    default_metric = 'ad exchange ad requests' if 'ad exchange ad requests' in metricas_disponiveis else metricas_disponiveis[0]
    selected_metric = st.selectbox(
        f"Selecione a métrica para o gráfico e análise de crescimento:",
        options=metricas_disponiveis,
        index=metricas_disponiveis.index(default_metric),
        key=f"metric_growth_{key_prefix}"
    )

    metrics = calculate_metrics_safely(df, group_cols, metrics_config)
    
    if not metrics.empty:
        st.write(f"**{title}:**")
        
        # Formatação da tabela
        format_dict = {}
        for col in metrics.columns:
            if col in group_cols:
                if col == 'data':
                    format_dict[col] = lambda x: x.strftime('%d/%m/%Y') if pd.notnull(x) else ''
                continue  # Pula outras colunas de dimensão
                
            if pd.api.types.is_numeric_dtype(metrics[col]):
                if 'ad requests' in col.lower():
                    format_dict[col] = "{:,.0f}"
                elif any(x in col.lower() for x in ['rate', 'ctr', 'viewable']):
                    format_dict[col] = "{:.2f}%"
                elif any(x in col.lower() for x in ['ecpm', 'cpc', '$']):
                    format_dict[col] = "US$ {:.2f}"
                else:
                    format_dict[col] = "{:.2f}"
        
        # Aplicar formatação apenas nas colunas numéricas
        styled_metrics = metrics.style.format(format_dict)
        
        # Aplicar gradiente apenas nas colunas numéricas
        numeric_cols = [col for col in metrics.columns if pd.api.types.is_numeric_dtype(metrics[col])]
        if numeric_cols:
            styled_metrics = styled_metrics.background_gradient(
                cmap="RdYlGn",
                subset=numeric_cols
            )
        
        st.dataframe(styled_metrics, use_container_width=True)
        
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
            else:
                group_by_cols = ['data'] + [col for col in group_cols if col != 'data']
                color_col = group_cols[1] if len(group_cols) > 1 else group_cols[0]
                df_graph = df.copy()
            
            # Escolher agregação correta para a métrica selecionada
            metricas_soma = [
                'ad exchange ad requests',
                'ad exchange revenue ($)'
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
                
                # Adicionar métricas de crescimento
                if len(trend['data'].unique()) > 1:
                    st.subheader("📈 Análise de Crescimento")
                    
                    # Remover linhas de total se existirem
                    if 'Grupo' in trend.columns and 'TOTAL GERAL' in trend['Grupo'].values:
                        trend = trend[trend['Grupo'] != 'TOTAL GERAL']
                    if color_col in trend.columns and 'TOTAL GERAL' in trend[color_col].values:
                        trend = trend[trend[color_col] != 'TOTAL GERAL']
                    
                    # Calcular crescimento
                    growth_data = []
                    for group in trend[color_col].unique():
                        group_data = trend[trend[color_col] == group].copy()
                        # Só calcula crescimento se houver pelo menos 2 datas distintas
                        if len(group_data['data'].unique()) > 1:
                            group_data = group_data.sort_values('data')
                            first_value = group_data[selected_metric].iloc[0]
                            last_value = group_data[selected_metric].iloc[-1]
                            growth_pct = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                            
                            # Adicionar informações específicas baseadas no tipo de análise
                            growth_info = {
                                'Grupo': group,
                                'Valor Inicial': first_value,
                                'Valor Final': last_value,
                                'Crescimento Absoluto': last_value - first_value,
                                'Crescimento %': growth_pct
                            }
                            
                            # Adicionar advertiser se estiver na aba de advertiser
                            if 'advertiser (classified)' in df.columns and key_prefix == 'advertiser':
                                advertiser = df[df[color_col] == group]['advertiser (classified)'].iloc[0]
                                growth_info['Advertiser'] = advertiser
                            
                            growth_data.append(growth_info)
                    
                    if growth_data:
                        growth_df = pd.DataFrame(growth_data)
                        
                        # Reordenar colunas para mostrar Advertiser logo após o Grupo
                        if 'Advertiser' in growth_df.columns:
                            cols = growth_df.columns.tolist()
                            cols.remove('Advertiser')
                            cols.insert(1, 'Advertiser')
                            growth_df = growth_df[cols]
                        
                        growth_df = growth_df.sort_values('Crescimento %', ascending=False)
                        
                        # Formatar tabela de crescimento
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
        if st.button(f"📤 Exportar dados de {title}", key=f"export_{key_prefix}"):
            export_cols = group_cols.copy()
            export_cols.extend(metrics_config.keys())
            
            csv_data = metrics[export_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                f"📥 Baixar CSV - {title}",
                csv_data,
                f"analise_{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                key=f"download_{key_prefix}"
            )
    else:
        st.warning("⚠️ Nenhum dado encontrado com a configuração atual.")

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

def create_metric_selector(metricas_disponiveis, key_prefix):
    """Cria um seletor de métricas"""
    return st.multiselect(
        "Métricas para exibir:",
        options=metricas_disponiveis,
        default=metricas_disponiveis[:3],
        key=f"metricas_{key_prefix}"
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
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file.name)
            all_data.append(df)
        
        df_original = pd.concat(all_data, ignore_index=True)
        
        # Padronização
        df_original.columns = [c.strip().lower() for c in df_original.columns]
        df_original.rename(columns={'channel': 'utm_source', 'date': 'data'}, inplace=True)
        df_original['data'] = pd.to_datetime(df_original['data'], errors='coerce')
        
        # Processamento numérico
        metricas_numericas = [
            'ad exchange ad requests',
            'ad exchange match rate', 
            'ad exchange ad request ecpm ($)',
            'ad exchange cpc ($)',
            'ad exchange ctr',
            'ad exchange active view % viewable impressions'
        ]
        
        for coluna in metricas_numericas:
            if coluna in df_original.columns:
                df_original[coluna] = pd.to_numeric(
                    df_original[coluna].astype(str).str.replace('%', '').str.replace(',', '').str.strip(),
                    errors='coerce'
                )
        
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
        # Filtro de utm_source
        todas_sources = sorted(df_original['utm_source'].dropna().unique())
        selected_sources = st.multiselect(
            "🌐 Canais (utm_source):",
            options=todas_sources,
            default=todas_sources[:5] if len(todas_sources) <= 5 else todas_sources[:3]
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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Visão Geral", 
        "🔗 Por URL", 
        "📱 Por Ad Type", 
        "🏢 Por Advertiser", 
        "📦 Por Ad Unit"
    ])
    
    with tab1:
        st.subheader("📋 Consolidado por UTM Source")
        
        # Métricas consolidadas
        metrics_to_agg = {m: metricas_config[m] for m in metricas_disponiveis}
        consolidado = calculate_metrics_safely(
            df_filtered, 
            ['utm_source'], 
            metrics_to_agg
        )
        
        # Adicionar total
        if not consolidado.empty:
            total_row = {}
            for col in metricas_disponiveis:
                if metricas_config[col] == 'sum':
                    total_row[col] = consolidado[col].sum()
                else:
                    # Para médias, calcular média ponderada se possível
                    if 'ad exchange ad requests' in consolidado.columns:
                        weights = consolidado['ad exchange ad requests']
                        total_row[col] = np.average(consolidado[col], weights=weights)
                    else:
                        total_row[col] = consolidado[col].mean()
            
            total_row['utm_source'] = 'TOTAL GERAL'
            total_df = pd.DataFrame([total_row])
            consolidado = pd.concat([consolidado, total_df], ignore_index=True)
        
        # Renomear colunas para exibição
        display_columns = {
            'ad exchange ad requests': 'Ad Requests',
            'ad exchange match rate': 'Match Rate (%)',
            'ad exchange ad request ecpm ($)': 'eCPM (US$)',
            'ad exchange cpc ($)': 'CPC (US$)',
            'ad exchange ctr': 'CTR (%)',
            'ad exchange active view % viewable impressions': 'Viewability (%)'
        }
        
        consolidado_display = consolidado.rename(columns=display_columns)
        
        # Formatação
        format_dict = {}
        for col in consolidado_display.columns:
            if col == 'utm_source':
                continue  # Pula coluna de texto
                
            if pd.api.types.is_numeric_dtype(consolidado_display[col]):
                if 'Ad Requests' in col:
                    format_dict[col] = "{:,.0f}"
                elif 'Match Rate (%)' in col or 'CTR (%)' in col or 'Viewability (%)' in col:
                    format_dict[col] = "{:.2f}%"
                elif 'eCPM (US$)' in col or 'CPC (US$)' in col:
                    format_dict[col] = "US$ {:.2f}"
                else:
                    format_dict[col] = "{:.2f}"
        
        st.dataframe(
            consolidado_display.style.format(format_dict),
            use_container_width=True
        )
        
        # Gráfico de tendência
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
    
    with tab2:
        if 'url' not in df_filtered.columns:
            st.warning("⚠️ Coluna 'URL' não encontrada nos dados.")
        else:
            st.subheader("🔗 Análise por URL")
            
            # Top URLs por Ad Requests
            if 'ad exchange ad requests' in df_filtered.columns:
                top_urls_data = calculate_metrics_safely(
                    df_filtered,
                    ['url'],
                    {'ad exchange ad requests': 'sum'}
                )
                
                if not top_urls_data.empty:
                    top_urls_data = top_urls_data.sort_values('ad exchange ad requests', ascending=False).head(20)
                    
                    st.write("**Top 20 URLs por Ad Requests:**")
                    st.dataframe(
                        top_urls_data.style.format({'ad exchange ad requests': "{:,.0f}"}),
                        use_container_width=True
                    )
                    
                    # Filtro de URLs para análise detalhada
                    urls_para_analise = st.multiselect(
                        "Selecione URLs para análise detalhada:",
                        options=top_urls_data['url'].tolist(),
                        default=top_urls_data['url'].head(3).tolist()
                    )
                    
                    if urls_para_analise:
                        df_urls = df_filtered[df_filtered['url'].isin(urls_para_analise)]
                        
                        # Métricas por URL e UTM Source
                        url_metrics = calculate_metrics_safely(
                            df_urls,
                            ['url', 'utm_source'],
                            {m: metricas_config[m] for m in metricas_disponiveis}
                        )
                        
                        if not url_metrics.empty:
                            st.subheader("📊 Métricas Detalhadas por URL")
                            st.dataframe(url_metrics, use_container_width=True)
                            
                            # Gráfico de eCPM por URL
                            if 'ad exchange ad request ecpm ($)' in url_metrics.columns:
                                fig_ecpm = px.bar(
                                    url_metrics,
                                    x='url',
                                    y='ad exchange ad request ecpm ($)',
                                    color='utm_source',
                                    title='eCPM por URL e UTM Source'
                                )
                                fig_ecpm.update_xaxes(tickangle=45)
                                st.plotly_chart(fig_ecpm, use_container_width=True)
    
    with tab3:
        if 'ad type' not in df_filtered.columns:
            st.warning("⚠️ Coluna 'Ad Type' não encontrada nos dados.")
        else:
            st.subheader("📱 Análise por Ad Type")
            
            # Filtros
            col1, col2 = st.columns(2)
            
            with col1:
                additional_url_filter = create_url_filter(df_filtered, "adtype")
                selected_sources = create_source_filter(df_filtered, "adtype")
            
            with col2:
                st.write("**Filtros Adicionais:**")
                ad_types_disponiveis = sorted(df_filtered['ad type'].dropna().unique())
                selected_ad_types = st.multiselect(
                    "Ad Types:",
                    options=ad_types_disponiveis,
                    default=ad_types_disponiveis[:10] if len(ad_types_disponiveis) <= 10 else []
                )
            
            # Aplicar filtros
            additional_filters = {}
            if additional_url_filter:
                additional_filters['url'] = additional_url_filter
            if selected_ad_types:
                additional_filters['ad type'] = selected_ad_types
            
            df_adtype = get_filtered_data(df_filtered, date_range, selected_sources, additional_filters)
            df_adtype = df_adtype[~df_adtype['ad type'].str.lower().str.contains('unmatched ad requests', na=False)]
            
            if not df_adtype.empty:
                # Configuração
                col1, col2 = st.columns(2)
                
                with col1:
                    dimensoes = create_dimension_selector(df_adtype, ['ad type', 'utm_source'], "adtype")
                
                with col2:
                    metricas = create_metric_selector(metricas_disponiveis, "adtype")
                
                if dimensoes and metricas:
                    metrics_config = {m: metricas_config[m] for m in metricas}
                    create_analysis_section(df_adtype, dimensoes, metrics_config, "Métricas por Ad Type", "adtype")
            else:
                st.info("ℹ️ Nenhum dado válido de Ad Type encontrado após filtros.")
    
    with tab4:
        if 'advertiser (classified)' not in df_filtered.columns:
            st.warning("⚠️ Coluna 'Advertiser (classified)' não encontrada nos dados.")
        else:
            st.subheader("🏢 Análise por Advertiser")
            
            # Filtros
            col1, col2 = st.columns(2)
            
            with col1:
                additional_url_filter = create_url_filter(df_filtered, "advertiser")
                selected_sources = create_source_filter(df_filtered, "advertiser")
            
            with col2:
                st.write("**Filtros Adicionais:**")
                advertisers_disponiveis = sorted(df_filtered['advertiser (classified)'].dropna().unique())
                selected_advertisers = st.multiselect(
                    "Advertisers:",
                    options=advertisers_disponiveis,
                    default=advertisers_disponiveis[:10] if len(advertisers_disponiveis) <= 10 else []
                )
            
            # Aplicar filtros
            additional_filters = {}
            if additional_url_filter:
                additional_filters['url'] = additional_url_filter
            if selected_advertisers:
                additional_filters['advertiser (classified)'] = selected_advertisers
            
            df_adv = get_filtered_data(df_filtered, date_range, selected_sources, additional_filters)
            df_adv = df_adv[~df_adv['advertiser (classified)'].str.lower().str.contains('unmatched ad requests', na=False)]
            
            if not df_adv.empty:
                # Configuração
                col1, col2 = st.columns(2)
                
                with col1:
                    dimensoes = create_dimension_selector(df_adv, ['advertiser (classified)', 'utm_source'], "advertiser")
                
                with col2:
                    metricas = create_metric_selector(metricas_disponiveis, "advertiser")
                
                if dimensoes and metricas:
                    metrics_config = {m: metricas_config[m] for m in metricas}
                    create_analysis_section(df_adv, dimensoes, metrics_config, "Métricas por Advertiser", "advertiser")
            else:
                st.info("ℹ️ Nenhum dado válido de Advertiser encontrado após filtros.")
    
    with tab5:
        if 'ad unit' not in df_filtered.columns:
            st.warning("⚠️ Coluna 'Ad Unit' não encontrada nos dados.")
        else:
            st.subheader("📦 Análise por Ad Unit")
            
            # Filtros
            col1, col2 = st.columns(2)
            
            with col1:
                additional_url_filter = create_url_filter(df_filtered, "adunit")
                selected_sources = create_source_filter(df_filtered, "adunit")
            
            with col2:
                st.write("**Filtros Adicionais:**")
                ad_units_disponiveis = sorted(df_filtered['ad unit'].dropna().unique())
                selected_ad_units = st.multiselect(
                    "Ad Units:",
                    options=ad_units_disponiveis,
                    default=ad_units_disponiveis[:10] if len(ad_units_disponiveis) <= 10 else []
                )
            
            # Aplicar filtros
            additional_filters = {}
            if additional_url_filter:
                additional_filters['url'] = additional_url_filter
            if selected_ad_units:
                additional_filters['ad unit'] = selected_ad_units
            
            df_ad_unit = get_filtered_data(df_filtered, date_range, selected_sources, additional_filters)
            
            if not df_ad_unit.empty:
                # Configuração
                col1, col2 = st.columns(2)
                
                with col1:
                    dimensoes = create_dimension_selector(df_ad_unit, ['ad unit', 'utm_source'], "adunit")
                
                with col2:
                    metricas = create_metric_selector(metricas_disponiveis, "adunit")
                
                if dimensoes and metricas:
                    metrics_config = {m: metricas_config[m] for m in metricas}
                    create_analysis_section(df_ad_unit, dimensoes, metrics_config, "Métricas por Ad Unit", "adunit")
            else:
                st.info("ℹ️ Selecione Ad Units para visualizar os dados.")
    
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
