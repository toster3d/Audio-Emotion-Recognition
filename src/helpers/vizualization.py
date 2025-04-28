import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Stała szerokość dla wszystkich wykresów
STANDARD_WIDTH = 1200

def generate_accuracy_comparison_plot(results_df, save_dir):
    """Generuje i zapisuje wykres porównujący dokładność i czas treningu."""
    # Wykres połączony - Plotly Subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Porównanie dokładności dla różnych reprezentacji audio', 
                      'Porównanie czasu treningu dla różnych reprezentacji audio')
    )
    
    # Dodanie słupków dokładności
    fig.add_trace(
        go.Bar(
            x=results_df['Feature Type'], 
            y=results_df['Test Accuracy (%)'],
            text=results_df['Test Accuracy (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            marker_color='purple',
            name='Dokładność'
        ),
        row=1, col=1
    )
    
    # Dodanie słupków czasu treningu
    fig.add_trace(
        go.Bar(
            x=results_df['Feature Type'], 
            y=results_df['Training Time (s)'],
            text=results_df['Training Time (s)'].apply(lambda x: f'{int(x)}s'),
            textposition='outside',
            marker_color='purple',
            name='Czas treningu'
        ),
        row=1, col=2
    )
    
    # Aktualizacja układu
    fig.update_layout(
        height=600,
        width=STANDARD_WIDTH,
        showlegend=False,
        template='plotly_white'
    )
    
    # Aktualizacja osi X i Y dla obu wykresów
    fig.update_xaxes(title_text='Typ cechy', tickangle=-45, row=1, col=1)
    fig.update_xaxes(title_text='Typ cechy', tickangle=-45, row=1, col=2)
    fig.update_yaxes(title_text='Dokładność testu (%)', row=1, col=1)
    fig.update_yaxes(title_text='Czas treningu (s)', row=1, col=2)
    
    # Zapisanie i wyświetlenie wykresu połączonego
    combined_path = os.path.join(save_dir, 'combined_comparison_auto.html')
    fig.write_html(combined_path)
    fig.show()
    
    return combined_path

def generate_emotion_visualizations(emotions_df, results_df, save_dir):
    """Generuje i zapisuje wizualizacje dotyczące emocji."""
    
    # 1. Heatmapa F1-score dla każdej emocji i typu cechy
    pivot_df = emotions_df.pivot(index='Feature Type', columns='Emotion', values='F1-score')
    
    # Sortuj wiersze według średniej wartości F1 (malejąco)
    pivot_df['mean'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('mean', ascending=False)
    pivot_df = pivot_df.drop(columns=['mean'])
    
    fig_heatmap = px.imshow(
        pivot_df,
        text_auto='.1f',
        aspect="auto",
        color_continuous_scale='Purples',
        title='F1-score dla każdej emocji i typu cechy (%)'
    )
    
    fig_heatmap.update_layout(
        xaxis_title='Emocja',
        yaxis_title='Typ cechy',
        coloraxis_colorbar_title='F1-score (%)',
        width=STANDARD_WIDTH,
        height=800
    )
    
    # Zapisanie i wyświetlenie heatmapy
    heatmap_path = os.path.join(save_dir, 'emotions_heatmap_auto.html')
    fig_heatmap.write_html(heatmap_path)
    fig_heatmap.show()
    
    # 2. Wykres słupkowy dla najlepszych 3 reprezentacji - porównanie F1-score dla każdej emocji
    top_path = None
    if results_df is not None and not results_df.empty:
        # Wybierz 3 najlepsze reprezentacje
        top_features = results_df.sort_values('Test Accuracy (%)', ascending=False).head(3)['Feature Type'].tolist()
        top_emotions_df = emotions_df[emotions_df['Feature Type'].isin(top_features)]
        
        # Średnia F1-score dla każdej emocji
        fig_top = px.bar(
            top_emotions_df,
            x='Emotion',
            y='F1-score',
            color='Feature Type',
            barmode='group',
            title='Porównanie F1-score dla 3 najlepszych reprezentacji',
            color_discrete_sequence=px.colors.sequential.Purples[2:5]
        )
        
        fig_top.update_layout(
            xaxis_title='Emocja',
            yaxis_title='F1-score (%)',
            legend_title='Typ cechy',
            width=STANDARD_WIDTH,
            height=600
        )
        
        # Zapisanie i wyświetlenie wykresu
        top_path = os.path.join(save_dir, 'top_features_emotions_auto.html')
        fig_top.write_html(top_path)
        fig_top.show()
    
    # 3. Radar chart dla każdej reprezentacji
    feature_types = emotions_df['Feature Type'].unique()
    n_features = len(feature_types)
    
    # Określ liczbę wierszy i kolumn
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig_radar = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=feature_types,
        specs=[[{'type': 'polar'}] * n_cols] * n_rows
    )
    
    # Dla każdej reprezentacji utwórz wykres radarowy
    for i, feature in enumerate(feature_types):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        feature_df = emotions_df[emotions_df['Feature Type'] == feature]
        
        # Dodaj wykres radarowy
        fig_radar.add_trace(
            go.Scatterpolar(
                r=feature_df['F1-score'].values,
                theta=feature_df['Emotion'].values,
                fill='toself',
                name=feature,
                line_color='purple'
            ),
            row=row, 
            col=col
        )
        
        # Ustaw zakres osi r
        fig_radar.update_layout(**{
            f'polar{i+1}': dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            )
        })
    
    # Aktualizacja układu
    radar_height = max(380 * n_rows, 600)
    fig_radar.update_layout(
        height=radar_height,
        width=STANDARD_WIDTH,
        showlegend=False,
    )
    
    # Zapisanie i wyświetlenie wykresów radarowych
    radar_path = os.path.join(save_dir, 'emotions_radar_auto.html')
    fig_radar.write_html(radar_path)
    fig_radar.show()
    
    # 4. Combined dashboard - wszystkie reprezentacje, każda emocja
    emotions = emotions_df['Emotion'].unique()
    
    # Utwórz dashboard z wykresami słupkowymi dla każdej emocji
    fig_dashboard = make_subplots(
        rows=3, 
        cols=2,
        subplot_titles=emotions,
        vertical_spacing=0.1
    )
    
    # Kolor dla każdej emocji
    emotion_colors = {
        'anger': 'red',
        'fear': 'purple',
        'happiness': 'yellow',
        'neutral': 'blue',
        'sadness': 'gray',
        'surprised': 'green'
    }
    
    # Dla każdej emocji dodaj wykres
    for i, emotion in enumerate(emotions):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Wybierz dane dla danej emocji
        emotion_data = emotions_df[emotions_df['Emotion'] == emotion]
        # Sortuj według F1-score
        emotion_data = emotion_data.sort_values('F1-score', ascending=False)
        
        # Wybierz kolor lub użyj domyślnego fioletowego
        color = emotion_colors.get(emotion.lower(), 'purple')
        
        # Dodaj wykres słupkowy
        fig_dashboard.add_trace(
            go.Bar(
                x=emotion_data['Feature Type'],
                y=emotion_data['F1-score'],
                marker_color=color,
                text=emotion_data['F1-score'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside'
            ),
            row=row, 
            col=col
        )
        
        # Ustaw tytuły osi
        fig_dashboard.update_yaxes(title_text='F1-score (%)', range=[0, 110], row=row, col=col)
        fig_dashboard.update_xaxes(tickangle=-45, row=row, col=col)
    
    # Aktualizacja układu
    fig_dashboard.update_layout(
        height=1000,
        width=STANDARD_WIDTH,
        showlegend=False,
        title_text='Porównanie F1-score dla różnych emocji według typu reprezentacji'
    )
    
    # Zapisanie i wyświetlenie dashboardu
    dashboard_path = os.path.join(save_dir, 'emotions_dashboard_auto.html')
    fig_dashboard.write_html(dashboard_path)
    fig_dashboard.show()
    
    return {
        'heatmap': heatmap_path,
        'top_features': top_path,
        'radar': radar_path,
        'dashboard': dashboard_path
    }