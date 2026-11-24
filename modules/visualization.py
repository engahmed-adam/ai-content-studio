"""Charts and HTML fragments for the dashboard."""
from __future__ import annotations

import html

import plotly.graph_objects as go

from config import AI_THRESHOLD, HUMAN_THRESHOLD

GREEN, AMBER, RED = "#2ea043", "#d29922", "#f85149"


def _color(p: float) -> str:
    return RED if p >= AI_THRESHOLD else GREEN if p <= HUMAN_THRESHOLD else AMBER


def gauge(prob: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob * 100, 1), number={"suffix": "%"},
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": _color(prob)},
               "steps": [{"range": [0, 30], "color": "rgba(46,160,67,.18)"},
                         {"range": [30, 70], "color": "rgba(210,153,34,.18)"},
                         {"range": [70, 100], "color": "rgba(248,81,73,.18)"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def segment_bar(segments) -> go.Figure:
    ys = [round(s.ai_prob * 100, 1) for s in segments]
    fig = go.Figure(go.Bar(x=list(range(1, len(ys) + 1)), y=ys, marker_color=[_color(s.ai_prob) for s in segments]))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 100], title="AI %"),
                      xaxis_title="Passage", title="AI likelihood per passage", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def highlight_html(segments) -> str:
    out = []
    for s in segments:
        alpha = 0.10 + 0.45 * s.ai_prob if s.ai_prob >= 0.5 else 0.0
        bg = f"rgba(248,81,73,{alpha:.2f})" if alpha else "transparent"
        out.append(f'<span title="AI likelihood {s.ai_prob:.0%}" style="background:{bg};padding:2px 3px;'
                   f'border-radius:4px;line-height:1.9">{html.escape(s.text)}</span>')
    return '<div style="font-size:0.95rem">' + " ".join(out) + "</div>"
