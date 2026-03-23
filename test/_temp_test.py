# interactive_plotly_demos.py
# Requirements: pip install plotly

import shutil
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(fig, filename):
    """Helper to save plots with small offline-friendly size"""
    fig.write_html(
        str(filename),
        include_plotlyjs="cdn",  # ~60–80 KB + data
        full_html=True,
        config={"displayModeBar": True},
    )
    print(f"Saved: {filename}")


def demo_1_scatter():
    """Simple scatter plot with size, color and hover"""
    data = {
        "x": [5.1, 4.9, 4.7, 4.6, 5.0, 7.0, 6.4, 6.9, 5.5, 6.5],
        "y": [3.5, 3.0, 3.2, 3.1, 3.6, 3.2, 3.2, 3.1, 2.3, 2.8],
        "size": [14, 15, 13, 15, 16, 51, 45, 57, 33, 39],
        "category": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "label": ["setosa"] * 5 + ["versicolor"] * 5,
    }
    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        size="size",
        color="category",
        hover_data=["label", "size"],
        title="Demo 1 – Basic Scatter with size & color",
        labels={"x": "Feature 1", "y": "Feature 2"},
        opacity=0.8,
    )

    fig.update_layout(width=700, height=500, showlegend=True)

    save_plot(fig, OUTPUT_DIR / "demo_1_scatter.html")
    fig.show()


def demo_2_line():
    """Line chart with multiple traces"""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
    sales_a = [120, 145, 170, 155, 190, 210, 195, 220]
    sales_b = [80, 95, 110, 130, 125, 150, 170, 185]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=months,
            y=sales_a,
            mode="lines+markers",
            name="Product A",
            line=dict(color="#636EFA", width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=months,
            y=sales_b,
            mode="lines+markers",
            name="Product B",
            line=dict(color="#EF553B", width=2.5),
        )
    )

    fig.update_layout(
        title="Demo 2 – Monthly Sales Comparison",
        xaxis_title="Month",
        yaxis_title="Revenue (k$)",
        hovermode="x unified",
        width=780,
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    save_plot(fig, OUTPUT_DIR / "demo_2_line.html")
    fig.show()


def demo_3_bar_grouped():
    """Grouped bar chart"""
    categories = ["Q1", "Q2", "Q3", "Q4"]
    team_a = [45, 52, 48, 60]
    team_b = [38, 41, 55, 49]
    team_c = [29, 35, 40, 52]

    fig = go.Figure(
        data=[
            go.Bar(name="Team A", x=categories, y=team_a, marker_color="#4e79a7"),
            go.Bar(name="Team B", x=categories, y=team_b, marker_color="#f28e2b"),
            go.Bar(name="Team C", x=categories, y=team_c, marker_color="#76b7b2"),
        ]
    )

    fig.update_layout(
        barmode="group",
        title="Demo 3 – Quarterly Performance – Grouped Bars",
        xaxis_title="Quarter",
        yaxis_title="Points",
        width=760,
        height=520,
        legend=dict(orientation="h", y=1.1),
    )

    save_plot(fig, OUTPUT_DIR / "demo_3_bar_grouped.html")
    fig.show()


def demo_4_boxplot():
    """Box plot + strip (jitter) points"""
    data = {
        "score": [
            78,
            85,
            92,
            88,
            76,
            81,
            95,
            89,
            72,
            84,
            91,
            87,
            79,
            83,
            96,
            65,
            71,
            68,
            74,
            70,
            62,
            67,
            73,
            69,
            64,
        ],
        "group": ["Control"] * 15 + ["Treatment"] * 10,
    }
    df = pd.DataFrame(data)

    fig = px.box(
        df,
        x="group",
        y="score",
        color="group",
        points="all",  # shows all points
        title="Demo 4 – Score Distribution – Box + Strip",
        labels={"score": "Test Score", "group": ""},
        category_orders={"group": ["Control", "Treatment"]},
    )

    fig.update_traces(boxmean=True, marker=dict(opacity=0.7))
    fig.update_layout(width=680, height=520, showlegend=False)

    save_plot(fig, OUTPUT_DIR / "demo_4_box.html")
    fig.show()


def demo_5_pie_donut():
    """Pie / Donut chart"""
    labels = ["Product A", "Product B", "Product C", "Product D", "Other"]
    values = [38, 27, 19, 11, 5]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,  # makes it a donut
                textinfo="label+percent",
                insidetextorientation="radial",
                marker=dict(colors=px.colors.qualitative.Pastel),
            )
        ]
    )

    fig.update_layout(
        title="Demo 5 – Market Share (Donut)",
        width=680,
        height=520,
        annotations=[dict(text="2025", x=0.5, y=0.5, font_size=20, showarrow=False)],
    )

    save_plot(fig, OUTPUT_DIR / "demo_5_donut.html")
    fig.show()


# ────────────────────────────────────────
# Run selected / all demos
# ────────────────────────────────────────

if __name__ == "__main__":
    demo_1_scatter()
    demo_2_line()
    demo_3_bar_grouped()
    demo_4_boxplot()
    demo_5_pie_donut()
    print("\nAll demo plots saved as HTML files.")
