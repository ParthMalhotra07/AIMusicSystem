"""
Interactive Dashboard Module
Creates comprehensive HTML dashboard for all visualizations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class InteractiveDashboard:
    """
    Creates interactive HTML dashboard with all visualizations.
    """

    def __init__(self):
        """Initialize dashboard."""
        self.sections = []

    @staticmethod
    def create_embedding_scatter(
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        song_names: Optional[List[str]] = None,
        title: str = "Embedding Space"
    ):
        """
        Create interactive scatter plot of embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            2D projected embeddings (N x 2)
        labels : np.ndarray, optional
            Cluster labels
        song_names : list, optional
            Song names for hover
        title : str
            Plot title

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly not installed. Run: pip install plotly")

        if song_names is None:
            song_names = [f"Song {i}" for i in range(len(embeddings))]

        if labels is None:
            labels = np.zeros(len(embeddings), dtype=int)

        fig = go.Figure()

        for label in np.unique(labels):
            mask = labels == label
            label_name = f"Cluster {label}" if label >= 0 else "Noise"

            fig.add_trace(go.Scatter(
                x=embeddings[mask, 0],
                y=embeddings[mask, 1],
                mode='markers',
                name=label_name,
                text=[song_names[i] for i in np.where(mask)[0]],
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                marker=dict(
                    size=10,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                )
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            hovermode='closest',
            template='plotly_white',
            width=800,
            height=600
        )

        return fig

    @staticmethod
    def create_cluster_pie(
        labels: np.ndarray,
        title: str = "Cluster Distribution"
    ):
        """
        Create interactive pie chart of cluster sizes.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments
        title : str
            Plot title

        Returns
        -------
        plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly not installed. Run: pip install plotly")

        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        labels_text = [f"Cluster {label}" for label in unique_labels]

        fig = go.Figure(data=[go.Pie(
            labels=labels_text,
            values=counts,
            hole=0.3,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Songs: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])

        fig.update_layout(
            title=title,
            template='plotly_white',
            width=600,
            height=500
        )

        return fig

    @staticmethod
    def create_cluster_bar(
        labels: np.ndarray,
        title: str = "Cluster Sizes"
    ):
        """
        Create interactive bar chart of cluster sizes.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments
        title : str
            Plot title

        Returns
        -------
        plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly not installed. Run: pip install plotly")

        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

        fig = go.Figure(data=[go.Bar(
            x=[f"Cluster {label}" for label in unique_labels],
            y=counts,
            text=counts,
            textposition='auto',
            marker=dict(
                color='steelblue',
                line=dict(color='black', width=1)
            ),
            hovertemplate='<b>%{x}</b><br>Songs: %{y}<extra></extra>'
        )])

        fig.update_layout(
            title=title,
            xaxis_title="Cluster",
            yaxis_title="Number of Songs",
            template='plotly_white',
            width=700,
            height=500
        )

        return fig

    @staticmethod
    def create_metrics_comparison(
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "Method Comparison"
    ):
        """
        Create interactive bar chart comparing metrics.

        Parameters
        ----------
        metrics_dict : dict
            method_name -> metrics mapping
        title : str
            Plot title

        Returns
        -------
        plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly not installed. Run: pip install plotly")

        methods = list(metrics_dict.keys())
        metric_names = list(next(iter(metrics_dict.values())).keys())

        fig = go.Figure()

        for metric_name in metric_names:
            values = [metrics_dict[method].get(metric_name, 0) for method in methods]

            fig.add_trace(go.Bar(
                name=metric_name.replace('_', ' ').title(),
                x=methods,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' + metric_name + ': %{y:.4f}<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Method",
            yaxis_title="Score",
            barmode='group',
            template='plotly_white',
            width=900,
            height=500
        )

        return fig

    @staticmethod
    def create_feature_heatmap(
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 20,
        title: str = "Feature Distribution by Cluster"
    ):
        """
        Create interactive heatmap of features by cluster.

        Parameters
        ----------
        features : np.ndarray
            Features (N x D)
        labels : np.ndarray
            Cluster labels
        feature_names : list, optional
            Feature names
        top_k : int
            Number of top features to show
        title : str
            Plot title

        Returns
        -------
        plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly not installed. Run: pip install plotly")

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(features.shape[1])]

        unique_labels = sorted(set(labels[labels >= 0]))

        # Calculate mean features per cluster
        cluster_means = []
        for label in unique_labels:
            mask = labels == label
            cluster_mean = features[mask].mean(axis=0)
            cluster_means.append(cluster_mean)

        cluster_means = np.array(cluster_means)

        # Select top features by variance
        feature_variance = cluster_means.var(axis=0)
        top_idx = np.argsort(feature_variance)[-top_k:][::-1]

        data_to_plot = cluster_means[:, top_idx].T
        selected_names = [feature_names[i] for i in top_idx]

        fig = go.Figure(data=go.Heatmap(
            z=data_to_plot,
            x=[f"Cluster {l}" for l in unique_labels],
            y=selected_names,
            colorscale='RdYlGn',
            hovertemplate='Cluster: %{x}<br>Feature: %{y}<br>Value: %{z:.3f}<extra></extra>',
            colorbar=dict(title="Mean Value")
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Cluster",
            yaxis_title="Feature",
            template='plotly_white',
            width=800,
            height=600
        )

        return fig

    @staticmethod
    def create_song_table(
        song_names: List[str],
        labels: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        title: str = "Song Cluster Assignments"
    ):
        """
        Create interactive table of songs with cluster assignments.

        Parameters
        ----------
        song_names : list
            Song names
        labels : np.ndarray
            Cluster labels
        embeddings : np.ndarray, optional
            Embeddings for distance calculation
        title : str
            Table title

        Returns
        -------
        plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly not installed. Run: pip install plotly")

        # Prepare data
        cluster_strs = [f"Cluster {l}" if l >= 0 else "Noise" for l in labels]

        header = ['Song Name', 'Cluster']
        data = [song_names, cluster_strs]

        if embeddings is not None:
            # Calculate distance to cluster center
            unique_labels = sorted(set(labels[labels >= 0]))
            distances = []

            for i, label in enumerate(labels):
                if label >= 0:
                    mask = labels == label
                    center = embeddings[mask].mean(axis=0)
                    dist = np.linalg.norm(embeddings[i] - center)
                    distances.append(f"{dist:.2f}")
                else:
                    distances.append("N/A")

            header.append('Distance to Center')
            data.append(distances)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=header,
                fill_color='steelblue',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=data,
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])

        fig.update_layout(
            title=title,
            template='plotly_white',
            width=900,
            height=600
        )

        return fig

    @staticmethod
    def generate_dashboard(
        embeddings: np.ndarray,
        labels: np.ndarray,
        song_names: List[str],
        original_features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        comparison_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        title: str = "Music Clustering Dashboard",
        save_path: str = "dashboard.html"
    ):
        """
        Generate comprehensive HTML dashboard.

        Parameters
        ----------
        embeddings : np.ndarray
            2D projected embeddings (N x 2)
        labels : np.ndarray
            Cluster labels
        song_names : list
            Song names
        original_features : np.ndarray, optional
            Original high-dimensional features
        feature_names : list, optional
            Feature names
        metrics : dict, optional
            Clustering metrics
        comparison_metrics : dict, optional
            Metrics for multiple methods
        title : str
            Dashboard title
        save_path : str
            Path to save HTML file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Plotly not installed. Run: pip install plotly")

        print(f"🎨 Generating interactive dashboard...")

        # Create figures
        figures = []

        # 1. Embedding scatter
        print("   Creating embedding visualization...")
        fig_scatter = InteractiveDashboard.create_embedding_scatter(
            embeddings, labels, song_names, "Embedding Space (2D Projection)"
        )
        figures.append(("Embedding Space", fig_scatter))

        # 2. Cluster distribution (pie + bar)
        print("   Creating cluster distribution charts...")
        fig_pie = InteractiveDashboard.create_cluster_pie(labels, "Cluster Distribution")
        figures.append(("Cluster Pie Chart", fig_pie))

        fig_bar = InteractiveDashboard.create_cluster_bar(labels, "Cluster Sizes")
        figures.append(("Cluster Bar Chart", fig_bar))

        # 3. Song table
        print("   Creating song table...")
        fig_table = InteractiveDashboard.create_song_table(
            song_names, labels, embeddings, "Song Cluster Assignments"
        )
        figures.append(("Song Table", fig_table))

        # 4. Feature heatmap (if features provided)
        if original_features is not None:
            print("   Creating feature heatmap...")
            fig_heatmap = InteractiveDashboard.create_feature_heatmap(
                original_features, labels, feature_names, top_k=20
            )
            figures.append(("Feature Distribution", fig_heatmap))

        # 5. Metrics comparison (if provided)
        if comparison_metrics is not None:
            print("   Creating metrics comparison...")
            fig_metrics = InteractiveDashboard.create_metrics_comparison(
                comparison_metrics, "Method Comparison"
            )
            figures.append(("Method Comparison", fig_metrics))

        # Generate HTML
        print("   Assembling HTML dashboard...")
        html_content = InteractiveDashboard._generate_html(
            figures, title, metrics
        )

        # Save
        with open(save_path, 'w') as f:
            f.write(html_content)

        print(f"✓ Dashboard saved to {save_path}")
        print(f"  Open in browser to explore!")

    @staticmethod
    def _generate_html(
        figures: List[Tuple[str, any]],
        title: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Generate HTML content with all figures."""
        html_parts = []

        # HTML header
        html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .header {{
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            color: #333;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            color: #666;
            font-size: 1.1em;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .metrics {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metrics h2 {{
            margin-top: 0;
            color: #333;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .section {{
            background: white;
            margin: 20px 0;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .plot-container {{
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: white;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 {title}</h1>
        <p>Interactive Music Clustering Visualization</p>
    </div>
    <div class="container">
""")

        # Metrics section
        if metrics:
            html_parts.append("""
        <div class="metrics">
            <h2>📊 Clustering Metrics</h2>
            <div class="metric-grid">
""")
            for metric_name, value in metrics.items():
                display_name = metric_name.replace('_', ' ').title()
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)

                html_parts.append(f"""
                <div class="metric-card">
                    <h3>{display_name}</h3>
                    <div class="value">{value_str}</div>
                </div>
""")
            html_parts.append("""
            </div>
        </div>
""")

        # Add each figure
        for idx, (section_name, fig) in enumerate(figures):
            plot_div = fig.to_html(
                include_plotlyjs=False,
                div_id=f"plot_{idx}"
            )

            html_parts.append(f"""
        <div class="section">
            <h2>{section_name}</h2>
            <div class="plot-container">
                {plot_div}
            </div>
        </div>
""")

        # Footer
        html_parts.append("""
    </div>
    <div class="footer">
        <p>Generated by AI Music Recommendation System - Member 2: Embedding & Clustering</p>
        <p>Powered by Claude Code 🚀</p>
    </div>
</body>
</html>
""")

        return ''.join(html_parts)


if __name__ == '__main__':
    # Test dashboard generation
    print("Testing Interactive Dashboard...\n")

    np.random.seed(42)

    # Create synthetic data
    cluster1 = np.random.randn(20, 14) + 5
    cluster2 = np.random.randn(15, 14) - 5
    cluster3 = np.random.randn(10, 14)

    embeddings_hd = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0]*20 + [1]*15 + [2]*10)
    song_names = [f"song_{i:02d}" for i in range(len(embeddings_hd))]

    # Project to 2D for visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_hd)

    # Create synthetic metrics
    metrics = {
        'silhouette_score': 0.6523,
        'davies_bouldin_index': 0.8234,
        'n_clusters': 3
    }

    comparison_metrics = {
        'K-Means': {'silhouette_score': 0.6523, 'davies_bouldin_index': 0.8234},
        'DBSCAN': {'silhouette_score': 0.7123, 'davies_bouldin_index': 0.7456},
        'Hierarchical': {'silhouette_score': 0.5987, 'davies_bouldin_index': 0.9012}
    }

    # Generate dashboard
    InteractiveDashboard.generate_dashboard(
        embeddings=embeddings_2d,
        labels=labels,
        song_names=song_names,
        original_features=embeddings_hd,
        metrics=metrics,
        comparison_metrics=comparison_metrics,
        title="Test Music Dashboard",
        save_path="test_dashboard.html"
    )

    print("\n✓ Dashboard test complete")
