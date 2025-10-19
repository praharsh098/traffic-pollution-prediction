"""
Streamlit App for Traffic & Pollution Prediction
Interactive dashboard for exploring data and running predictions
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.data_processing import load_and_merge_two, clean_data
from src.feature_engineering import add_time_features, add_lag_features
from src.models import train_baseline_model
import os

# Page config
st.set_page_config(
    page_title="Traffic & Pollution Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(traffic_path: str, pollution_path: str):
    """Load and merge data with caching"""
    df = load_and_merge_two(traffic_path, pollution_path)
    df = clean_data(df)
    return df


def create_traffic_pollution_plot(df):
    """Create interactive plotly chart"""
    fig = go.Figure()
    
    # Add traffic volume trace
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['traffic_volume'],
        name='Traffic Volume',
        line=dict(color='#1f77b4', width=2),
        yaxis='y'
    ))
    
    # Add PM2.5 trace on secondary y-axis
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['pm25'],
        name='PM2.5',
        line=dict(color='#ff7f0e', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Traffic Volume & PM2.5 Trends Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Traffic Volume', side='left', color='#1f77b4'),
        yaxis2=dict(title='PM2.5 (μg/m³)', side='right', overlaying='y', color='#ff7f0e'),
        hovermode='x unified',
        height=500,
        legend=dict(x=0.7, y=1)
    )
    
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=600,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def create_distribution_plots(df):
    """Create distribution plots for traffic and pollution"""
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(
            df, 
            x='traffic_volume', 
            nbins=30,
            title='Traffic Volume Distribution',
            labels={'traffic_volume': 'Traffic Volume', 'count': 'Frequency'}
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.histogram(
            df, 
            x='pm25', 
            nbins=30,
            title='PM2.5 Distribution',
            labels={'pm25': 'PM2.5 (μg/m³)', 'count': 'Frequency'}
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)


def main():
    # Header
    st.markdown('<p class="main-header">🚗 Traffic & Pollution Prediction Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        st.subheader("📁 Data Upload")
        traffic_file = st.file_uploader(
            "Upload Traffic Data (CSV)",
            type=['csv'],
            key='traffic'
        )
        pollution_file = st.file_uploader(
            "Upload Pollution Data (CSV)",
            type=['csv'],
            key='pollution'
        )
        
        # Or use default files
        use_default = st.checkbox("Use Default Data Files", value=True)
        
        if use_default:
            traffic_path = "data/delhi_traffic.csv"
            pollution_path = "data/vehicle_emission.csv"
        else:
            traffic_path = None
            pollution_path = None
        
        st.divider()
        
        # Model parameters
        st.subheader("🤖 Model Parameters")
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        target_col = st.selectbox(
            "Target Variable",
            ["traffic_volume", "pm25"],
            index=0
        )
        
        st.divider()
        
        # Info
        st.info("💡 Upload CSV files with 'datetime', 'traffic_volume', and 'pm25' columns")
    
    # Main content
    if use_default or (traffic_file is not None and pollution_file is not None):
        try:
            # Load data
            with st.spinner("Loading data..."):
                if use_default:
                    df = load_data(traffic_path, pollution_path)
                else:
                    # Save uploaded files temporarily
                    temp_traffic = "temp_traffic.csv"
                    temp_pollution = "temp_pollution.csv"
                    
                    with open(temp_traffic, "wb") as f:
                        f.write(traffic_file.getbuffer())
                    with open(temp_pollution, "wb") as f:
                        f.write(pollution_file.getbuffer())
                    
                    df = load_data(temp_traffic, temp_pollution)
                    
                    # Clean up temp files
                    os.remove(temp_traffic)
                    os.remove(temp_pollution)
            
            # Display data info
            st.success(f"✅ Data loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🔍 Analysis", "🤖 Predictions"])
            
            with tab1:
                st.header("Data Overview")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Records",
                        f"{len(df):,}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Avg Traffic Volume",
                        f"{df['traffic_volume'].mean():.0f}",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Avg PM2.5",
                        f"{df['pm25'].mean():.1f}",
                        delta="μg/m³"
                    )
                
                with col4:
                    st.metric(
                        "Date Range",
                        f"{df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}",
                        delta=None
                    )
                
                st.divider()
                
                # Data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.subheader("Data Statistics")
                st.dataframe(df.describe(), use_container_width=True)
                
                st.subheader("Missing Values")
                missing = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing.index,
                    'Missing Count': missing.values,
                    'Percentage': (missing.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            
            with tab2:
                st.header("Trends & Patterns")
                
                # Time series plot
                st.plotly_chart(create_traffic_pollution_plot(df), use_container_width=True)
                
                # Distribution plots
                create_distribution_plots(df)
                
                # Hourly patterns
                st.subheader("Hourly Patterns")
                hourly_df = df.groupby(df['datetime'].dt.hour).agg({
                    'traffic_volume': 'mean',
                    'pm25': 'mean'
                }).reset_index()
                
                fig_hourly = go.Figure()
                fig_hourly.add_trace(go.Scatter(
                    x=hourly_df['datetime'],
                    y=hourly_df['traffic_volume'],
                    name='Avg Traffic Volume',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig_hourly.add_trace(go.Scatter(
                    x=hourly_df['datetime'],
                    y=hourly_df['pm25'],
                    name='Avg PM2.5',
                    line=dict(color='#ff7f0e', width=3),
                    yaxis='y2'
                ))
                fig_hourly.update_layout(
                    title='Average Traffic & Pollution by Hour of Day',
                    xaxis=dict(title='Hour', tickmode='linear', tick0=0, dtick=1),
                    yaxis=dict(title='Avg Traffic Volume', side='left'),
                    yaxis2=dict(title='Avg PM2.5 (μg/m³)', side='right', overlaying='y'),
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with tab3:
                st.header("Data Analysis")
                
                # Correlation heatmap
                st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
                
                # Feature engineering demo
                st.subheader("Feature Engineering Preview")
                if st.button("Add Time Features"):
                    df_eng = add_time_features(df.copy())
                    st.dataframe(df_eng[['datetime', 'hour', 'day_of_week', 'month', 'traffic_volume', 'pm25']].head(10), use_container_width=True)
            
            with tab4:
                st.header("Model Training & Predictions")
                
                st.info("🤖 Train a baseline Linear Regression model to predict traffic volume or pollution levels")
                
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        # Feature engineering
                        df_model = add_time_features(df.copy())
                        
                        # Select features
                        feature_cols = ['hour', 'day_of_week', 'month']
                        if 'traffic_volume' in df_model.columns and 'pm25' in df_model.columns:
                            if target_col == 'traffic_volume':
                                feature_cols.append('pm25')
                            else:
                                feature_cols.append('traffic_volume')
                        
                        # Train model
                        model = train_baseline_model(
                            df_model,
                            target_col=target_col,
                            feature_cols=feature_cols
                        )
                        
                        st.success("✅ Model trained successfully!")
                        
                        # Load predictions
                        predictions_path = "outputs/predictions.csv"
                        if os.path.exists(predictions_path):
                            pred_df = pd.read_csv(predictions_path)
                            
                            st.subheader("Predictions vs Actual")
                            
                            # Plot predictions
                            fig_pred = go.Figure()
                            fig_pred.add_trace(go.Scatter(
                                x=pred_df['datetime'],
                                y=pred_df['actual'],
                                name='Actual',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            fig_pred.add_trace(go.Scatter(
                                x=pred_df['datetime'],
                                y=pred_df['predicted'],
                                name='Predicted',
                                line=dict(color='#ff7f0e', width=2, dash='dash')
                            ))
                            fig_pred.update_layout(
                                title=f'{target_col.replace("_", " ").title()} Predictions vs Actual',
                                xaxis=dict(title='Date'),
                                yaxis=dict(title=target_col.replace("_", " ").title()),
                                height=500,
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            # Display predictions table
                            st.subheader("Prediction Results")
                            st.dataframe(pred_df, use_container_width=True)
                            
                            # Download button
                            csv = pred_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions (CSV)",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👈 Please upload data files or use default data to get started!")
        
        # Show sample data structure
        st.subheader("Expected Data Format")
        st.markdown("""
        **Traffic Data (delhi_traffic.csv):**
        - `datetime`: Date/time column
        - `traffic_volume` (or `Vehicles`): Number of vehicles
        
        **Pollution Data (vehicle_emission.csv):**
        - `datetime`: Date/time column (optional if same length as traffic)
        - `pm25` (or `PM2.5`): PM2.5 emissions in μg/m³
        """)


if __name__ == "__main__":
    main()

