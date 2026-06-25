import plotly.graph_objects as go
import plotly.express as px

COLORS = {
    'positive': '#28a745',
    'negative': '#dc3545',
    'neutral':  '#ffc107',
    'roberta':  '#4472C4',
    'vader':    '#ED7D31',
}
VI = {'positive': 'Tích cực', 'negative': 'Tiêu cực', 'neutral': 'Trung lập'}


def score_bars(neg: float, neu: float, pos: float, title: str = '') -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[neg, neu, pos],
        y=['Tiêu cực', 'Trung lập', 'Tích cực'],
        orientation='h',
        marker_color=[COLORS['negative'], COLORS['neutral'], COLORS['positive']],
        text=[f'{v:.3f}' for v in (neg, neu, pos)],
        textposition='auto',
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        height=220,
        margin=dict(l=5, r=15, t=40, b=5),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def radar_chart(rob: dict, vader: dict) -> go.Figure:
    cats = ['Tiêu cực', 'Trung lập', 'Tích cực', 'Tiêu cực']
    fig = go.Figure()
    for result, name, color, fill in [
        (rob,   'RoBERTa', COLORS['roberta'], 'rgba(68,114,196,0.25)'),
        (vader, 'VADER',   COLORS['vader'],   'rgba(237,125,49,0.25)'),
    ]:
        vals = [result['negative'], result['neutral'], result['positive'], result['negative']]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill='toself',
            name=name, line_color=color, fillcolor=fill, line_width=2,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%')),
        showlegend=True,
        title=dict(text='So sánh hai mô hình', font=dict(size=14)),
        height=320,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation='h', y=-0.12),
    )
    return fig


def confusion_matrix(y_true, y_pred, title: str) -> go.Figure:
    from sklearn.metrics import confusion_matrix as cm_fn
    labs    = ['negative', 'neutral', 'positive']
    labs_vi = ['Tiêu cực', 'Trung lập', 'Tích cực']
    valid   = [(t, p) for t, p in zip(y_true, y_pred) if t in labs and p in labs]
    if not valid:
        return go.Figure()
    y_t, y_p = zip(*valid)
    mat = cm_fn(list(y_t), list(y_p), labels=labs)
    fig = px.imshow(
        mat, x=labs_vi, y=labs_vi,
        color_continuous_scale='Blues',
        text_auto=True, title=title,
        labels=dict(x='Dự đoán', y='Thực tế'),
        aspect='equal',
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_traces(textfont_size=14)
    return fig


def metric_bars(rob_m: dict, vader_m: dict) -> go.Figure:
    keys   = ['accuracy', 'precision', 'recall', 'f1_score']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig = go.Figure()
    for data, name, color in [
        (rob_m,   'RoBERTa', COLORS['roberta']),
        (vader_m, 'VADER',   COLORS['vader']),
    ]:
        y = [data.get(k, 0) for k in keys]
        fig.add_trace(go.Bar(
            name=name, x=labels, y=y, marker_color=color,
            text=[f'{v:.3f}' for v in y], textposition='outside',
        ))
    fig.update_layout(
        barmode='group', title='So sánh hiệu suất mô hình',
        yaxis=dict(range=[0, 1.15]), height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation='h', y=1.02),
    )
    return fig


def sentiment_distribution(counts: dict, title: str) -> go.Figure:
    names  = [VI.get(k, k) for k in counts]
    colors = [COLORS.get(k, '#999') for k in counts]
    fig = px.pie(
        values=list(counts.values()),
        names=names,
        title=title,
        color_discrete_sequence=colors,
        hole=0.35,
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=50, b=10))
    return fig
