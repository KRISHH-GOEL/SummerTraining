"""
Lesson 06 : Dash Analytics Dashboard

Topics Covered
---------------
1. Dash Introduction
2. Dash Layout
3. HTML Components
4. Core Components
5. Plotly Graphs
6. Dropdown
7. Callbacks
8. Inputs
9. Outputs
10. KPI Cards
11. Interactive Dashboard

Dataset
-------
Titanic Dataset

Author : Krish Goel
Repository : Summer Training
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import seaborn as sns
import plotly.express as px

from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output
)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = sns.load_dataset("titanic")

print("=" * 80)
print("DASH ANALYTICS DASHBOARD")
print("=" * 80)

print("\nDataset Loaded Successfully")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

# ==========================================================
# DATA PREPARATION
# ==========================================================

# Remove rows without required values for visualizations.

plot_df = df.dropna(
    subset=[
        "age",
        "fare",
        "class",
        "sex"
    ]
).copy()

# ==========================================================
# KPI VALUES
# ==========================================================

total_passengers = len(df)

survival_rate = (
    df["survived"].mean() * 100
)

average_fare = (
    df["fare"].mean()
)

average_age = (
    df["age"].mean()
)

# ==========================================================
# CREATE DASH APPLICATION
# ==========================================================

app = Dash(__name__)

# ==========================================================
# APPLICATION TITLE
# ==========================================================

app.title = "Titanic Analytics Dashboard"

# ==========================================================
# DASHBOARD LAYOUT
# ==========================================================

app.layout = html.Div(

    [

        # ==================================================
        # HEADER
        # ==================================================

        html.Div(

            [

                html.H1(
                    "Titanic Analytics Dashboard",
                    style={
                        "marginBottom": "5px"
                    }
                ),

                html.P(
                    "Interactive Exploratory Data Analysis "
                    "Dashboard"
                )

            ],

            style={
                "textAlign": "center",
                "padding": "20px"
            }

        ),

        # ==================================================
        # KPI SECTION
        # ==================================================

        html.Div(

            [

                # Total Passengers

                html.Div(

                    [

                        html.H4(
                            "Total Passengers"
                        ),

                        html.H2(
                            f"{total_passengers:,}"
                        )

                    ],

                    style={
                        "textAlign": "center",
                        "padding": "20px",
                        "borderRadius": "10px",
                        "flex": "1"
                    }

                ),

                # Survival Rate

                html.Div(

                    [

                        html.H4(
                            "Survival Rate"
                        ),

                        html.H2(
                            f"{survival_rate:.1f}%"
                        )

                    ],

                    style={
                        "textAlign": "center",
                        "padding": "20px",
                        "borderRadius": "10px",
                        "flex": "1"
                    }

                ),

                # Average Fare

                html.Div(

                    [

                        html.H4(
                            "Average Fare"
                        ),

                        html.H2(
                            f"${average_fare:.2f}"
                        )

                    ],

                    style={
                        "textAlign": "center",
                        "padding": "20px",
                        "borderRadius": "10px",
                        "flex": "1"
                    }

                ),

                # Average Age

                html.Div(

                    [

                        html.H4(
                            "Average Age"
                        ),

                        html.H2(
                            f"{average_age:.1f}"
                        )

                    ],

                    style={
                        "textAlign": "center",
                        "padding": "20px",
                        "borderRadius": "10px",
                        "flex": "1"
                    }

                )

            ],

            style={
                "display": "flex",
                "gap": "15px",
                "padding": "20px"
            }

        ),

        # ==================================================
        # FILTER SECTION
        # ==================================================

        html.Div(

            [

                html.H3(
                    "Dashboard Filters"
                ),

                html.Label(
                    "Passenger Class"
                ),

                dcc.Dropdown(

                    id="class-dropdown",

                    options=[

                        {
                            "label": "All Classes",
                            "value": "All"
                        },

                        {
                            "label": "First Class",
                            "value": "First"
                        },

                        {
                            "label": "Second Class",
                            "value": "Second"
                        },

                        {
                            "label": "Third Class",
                            "value": "Third"
                        }

                    ],

                    value="All",

                    clearable=False

                ),

                html.Br(),

                html.Label(
                    "Gender"
                ),

                dcc.Dropdown(

                    id="gender-dropdown",

                    options=[

                        {
                            "label": "All",
                            "value": "All"
                        },

                        {
                            "label": "Male",
                            "value": "male"
                        },

                        {
                            "label": "Female",
                            "value": "female"
                        }

                    ],

                    value="All",

                    clearable=False

                )

            ],

            style={
                "padding": "20px"
            }

        ),

        # ==================================================
        # FIRST ROW OF CHARTS
        # ==================================================

        html.Div(

            [

                dcc.Graph(
                    id="survival-chart"
                ),

                dcc.Graph(
                    id="fare-chart"
                )

            ],

            style={
                "display": "flex",
                "gap": "20px"
            }

        ),

        # ==================================================
        # SECOND ROW OF CHARTS
        # ==================================================

        html.Div(

            [

                dcc.Graph(
                    id="age-chart"
                ),

                dcc.Graph(
                    id="class-chart"
                )

            ],

            style={
                "display": "flex",
                "gap": "20px"
            }

        )

    ],

    style={
        "fontFamily": "Arial, sans-serif",
        "padding": "10px"
    }

)

# ==========================================================
# CALLBACK
# ==========================================================

@app.callback(

    [

        Output(
            "survival-chart",
            "figure"
        ),

        Output(
            "fare-chart",
            "figure"
        ),

        Output(
            "age-chart",
            "figure"
        ),

        Output(
            "class-chart",
            "figure"
        )

    ],

    [

        Input(
            "class-dropdown",
            "value"
        ),

        Input(
            "gender-dropdown",
            "value"
        )

    ]

)

def update_dashboard(
    selected_class,
    selected_gender
):

    # ======================================================
    # FILTER DATA
    # ======================================================

    filtered_df = plot_df.copy()

    # Filter by passenger class

    if selected_class != "All":

        filtered_df = filtered_df[
            filtered_df["class"]
            == selected_class
        ]

    # Filter by gender

    if selected_gender != "All":

        filtered_df = filtered_df[
            filtered_df["sex"]
            == selected_gender
        ]

    # ======================================================
    # SURVIVAL CHART
    # ======================================================

    survival_data = (

        filtered_df
        .groupby(
            "class",
            as_index=False
        )["survived"]
        .mean()

    )

    survival_fig = px.bar(

        survival_data,

        x="class",

        y="survived",

        title="Survival Rate by Passenger Class",

        labels={
            "class": "Passenger Class",
            "survived": "Survival Rate"
        },

        text="survived"

    )

    survival_fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside"
    )

    survival_fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1]
    )

    survival_fig.update_layout(
        template="plotly_white"
    )

    # ======================================================
    # FARE DISTRIBUTION
    # ======================================================

    fare_fig = px.histogram(

        filtered_df,

        x="fare",

        color="class",

        nbins=30,

        title="Fare Distribution",

        labels={
            "fare": "Fare",
            "class": "Passenger Class"
        }

    )

    fare_fig.update_layout(
        template="plotly_white"
    )

    # ======================================================
    # AGE DISTRIBUTION
    # ======================================================

    age_fig = px.histogram(

        filtered_df,

        x="age",

        color="sex",

        nbins=25,

        marginal="box",

        title="Age Distribution",

        labels={
            "age": "Age",
            "sex": "Gender"
        }

    )

    age_fig.update_layout(
        template="plotly_white"
    )

    # ======================================================
    # CLASS DISTRIBUTION
    # ======================================================

    class_data = (

        filtered_df
        .groupby(
            "class",
            as_index=False
        )
        .size()

    )

    class_fig = px.pie(

        class_data,

        names="class",

        values="size",

        hole=0.4,

        title="Passenger Class Distribution"

    )

    class_fig.update_layout(
        template="plotly_white"
    )

    # ======================================================
    # RETURN FIGURES
    # ======================================================

    return (

        survival_fig,

        fare_fig,

        age_fig,

        class_fig

    )


# ==========================================================
# APPLICATION START
# ==========================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)

    print(
        "Dashboard starting..."
    )

    print(
        "Open http://127.0.0.1:8050/ in your browser."
    )

    print("=" * 80)

    app.run(
        debug=True
    )
