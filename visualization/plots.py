import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

class Visualizer:
    @staticmethod
    def plot_stock_trend(df: pd.DataFrame, ticker: str):
        """Plot real (unscaled) stock price with technical indicators."""
        st.write(f"### 📈 {ticker} Stock Price & Technical Indicators")

        # Use _Orig columns if available (unscaled), fall back to raw column
        close  = df.get('Close_Orig',  df.get('Close'))
        ma     = df.get('MA_14_Orig',  df.get('MA_14'))
        bb_hi  = df.get('BB_High_Orig', df.get('BB_High'))
        bb_lo  = df.get('BB_Low_Orig',  df.get('BB_Low'))

        x_data = df.index  # DatetimeIndex from yfinance

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data, y=close,  mode='lines', name='Close Price',
                                 line=dict(color='#00b4d8', width=2)))
        if ma is not None:
            fig.add_trace(go.Scatter(x=x_data, y=ma, mode='lines', name='MA-14',
                                     line=dict(dash='dash', color='#ffd166', width=1.5)))
        if bb_hi is not None:
            fig.add_trace(go.Scatter(x=x_data, y=bb_hi, mode='lines', name='BB Upper',
                                     line=dict(dash='dot', color='rgba(255,80,80,0.6)', width=1)))
        if bb_lo is not None:
            fig.add_trace(go.Scatter(x=x_data, y=bb_lo, mode='lines', name='BB Lower',
                                     fill='tonexty', fillcolor='rgba(0,180,216,0.05)',
                                     line=dict(dash='dot', color='rgba(80,255,80,0.6)', width=1)))

        fig.update_layout(xaxis_title='Date', yaxis_title='Price (USD)',
                          showlegend=True, height=400,
                          plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                          font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.Index):
        """Scatter plot of predicted vs actual trend labels."""
        st.write("### 🎯 Predicted vs Actual Trend  (-1 = Down | 0 = Neutral | 1 = Up)")

        label_map = {1: 'Uptrend', 0: 'Neutral', -1: 'Downtrend'}
        dates_list = list(dates)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates_list, y=y_true,
            mode='markers', name='Actual',
            marker=dict(symbol='circle', size=8, color='#06d6a0')))
        fig.add_trace(go.Scatter(
            x=dates_list, y=y_pred,
            mode='markers', name='Predicted',
            marker=dict(symbol='x-thin-open', size=10, color='#ef476f', line=dict(width=2))))

        fig.update_layout(
            xaxis_title='Date', yaxis_title='Trend',
            yaxis=dict(tickvals=[-1, 0, 1], ticktext=['📉 Down', '➡️ Neutral', '📈 Up']),
            showlegend=True, height=350,
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame):
        """Correlation heatmap — limited to meaningful feature columns only."""
        st.write("### 🔥 Feature Correlation Heatmap")

        # Only keep indicator columns (skip _Orig, _Fuzzy_ etc.)
        keep_cols = [c for c in df.columns
                     if not c.endswith('_Orig') and not c.startswith('Fuzzy')
                     and 'Fuzzy' not in c and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        numeric_df = df[keep_cols].select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            st.info("Not enough numeric columns to generate heatmap.")
            return

        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax, linewidths=0.3,
                    cbar_kws={'shrink': 0.8})
        ax.tick_params(colors='white', labelsize=7)
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_metrics_comparison(metrics_dict: dict):
        """Grouped bar chart comparing model metrics."""
        st.write("### 📊 Model Performance Comparison")

        if not metrics_dict:
            st.info("No metrics to display.")
            return

        df_metrics = pd.DataFrame(metrics_dict).T
        key_metrics = [c for c in ['Accuracy', 'Precision', 'Recall', 'F1 Score'] if c in df_metrics.columns]

        fig = go.Figure()
        colors = ['#00b4d8', '#06d6a0', '#ffd166', '#ef476f']
        for i, metric in enumerate(key_metrics):
            fig.add_trace(go.Bar(
                x=df_metrics.index, y=df_metrics[metric],
                name=metric, marker_color=colors[i % len(colors)]))

        fig.update_layout(
            barmode='group', yaxis_title='Score', showlegend=True,
            height=350, plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
