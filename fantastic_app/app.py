import pandas as pd
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pickle


# import/load trained models
model1 = pickle.load(open("model_1.pkl", "rb"))
model2 = pickle.load(open("model_2.pkl", "rb"))
model3 = pickle.load(open("model_3.pkl", "rb"))
model4 = pickle.load(open("model_4.pkl", "rb"))
model5 = pickle.load(open("model_5.pkl", "rb"))
model6 = pickle.load(open("model_6.pkl", "rb"))
xref_table = pd.read_csv("xref_table.csv")


# define a function to return the selected branches into a table
def generate_table(df):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"id": c, "name": c} for c in df.columns],
        style_cell={"textAlign": "center"},
        style_data={"border": "1px solid black"},
        style_header={
            "border": "1px solid black",
            "backgroundColor": "gray",
            "fontWeight": "bold",
        },
        style_as_list_view=True,
    )


# build the app
app = dash.Dash(__name__)
server = app.server


# build the app layout
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H2(
                    children="360 Discovery: Empowering Deeper Discovery with AI"
                ),
            ],
            style={"textAlign": "center", "border": "1px solid black"},
        ),
        html.Div(
            children=[
                html.H3(children="Client's Information: "),
                html.Div(
                    children=[
                        html.Label(
                            "1. Age: ",
                        ),
                        dcc.Input(
                            id="age",
                            type="text",
                            value="30",
                            style={"width": "150px"},
                        ),
                    ],
                    style={
                        "paddingRight": "30px",
                        "margin-bottom": "11px",
                        "font-size": "13px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    children=[
                        html.Label(
                            "2. Total Liquid Investable Assets: ",
                        ),
                        dcc.Input(
                            id="tlia",
                            type="text",
                            value="50000",
                            style={"width": "150px"},
                        ),
                    ],
                    style={
                        "paddingRight": "30px",
                        "margin-bottom": "11px",
                        "font-size": "13px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    children=[
                        html.Label(
                            "3. Share of Wallet: ",
                        ),
                        dcc.Input(
                            id="sow",
                            type="text",
                            value="0.2",
                            style={"width": "150px"},
                        ),
                    ],
                    style={
                        "paddingRight": "30px",
                        "margin-bottom": "11px",
                        "font-size": "13px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    children=[
                        html.Label(
                            "4. Living in Retirement: ",
                        ),
                        dcc.Input(
                            id="ret",
                            type="text",
                            value="0",
                            style={"width": "150px"},
                        ),
                    ],
                    style={
                        "paddingRight": "30px",
                        "margin-bottom": "11px",
                        "font-size": "13px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    children=[
                        html.Label(
                            "5. Kids: ",
                        ),
                        dcc.Input(
                            id="kid",
                            type="text",
                            value="1",
                            style={"width": "150px"},
                        ),
                    ],
                    style={
                        "paddingRight": "30px",
                        "margin-bottom": "11px",
                        "font-size": "13px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    children=[
                        html.Label(
                            "6. Estate Need:",
                        ),
                        dcc.Input(
                            id="est",
                            type="text",
                            value="0",
                            style={"width": "150px"},
                        ),
                    ],
                    style={
                        "paddingRight": "30px",
                        "margin-bottom": "11px",
                        "font-size": "13px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    children=[
                        html.Label(
                            "7. Wellness Score: ",
                        ),
                        dcc.Input(
                            id="well",
                            type="text",
                            value="50",
                            style={"width": "150px"},
                        ),
                    ],
                    style={
                        "paddingRight": "30px",
                        "margin-bottom": "11px",
                        "font-size": "13px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    children=[
                        html.Button(
                            "Submit",
                            id="submit_search",
                            style={
                                "paddingRight": "30px",
                                "margin-bottom": "11px",
                                "font-size": "13px",
                            },
                        ),
                    ],
                    style={
                        "display": "inline-block",
                    },
                ),
            ],
            style={
                "border": "1px solid black",
                "paddingLeft": "30px",
            },
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        dcc.Markdown(
                            id="output_1",
                            children="Selected Goals: ",
                        ),
                    ],
                    style={
                        "paddingRight": "30px",
                        "margin-bottom": "11px",
                        "font-size": "13px",
                    },
                ),
                html.Div(
                    id="output_2",
                ),
            ],
            style={
                # "width": "1200px",
                "border": "1px solid black",
                "paddingLeft": "30px",
            },
        ),
    ]
)


@app.callback(
    [
        Output("output_1", "children"),
        Output("output_2", "children"),
    ],
    [Input("submit_search", "n_clicks")],
    [
        State("age", "value"),
        State("tlia", "value"),
        State("sow", "value"),
        State("ret", "value"),
        State("kid", "value"),
        State("est", "value"),
        State("well", "value"),
    ],
)
def update_output(
    n_clicks,
    value1,
    value2,
    value3,
    value4,
    value5,
    value6,
    value7,
):

    age, tlia, sow, ret, kid, est, well = (
        float(value1),
        float(value2),
        float(value3),
        float(value4),
        float(value5),
        float(value6),
        float(value7),
    )

    # predict probabilities for scoring observation
    def score_func(
        age_temp,
        tlia_temp,
        sow_temp,
        ret_temp,
        kid_temp,
        est_temp,
        well_temp,
        model_1_temp,
        model_2_temp,
        model_3_temp,
        model_4_temp,
        model_5_temp,
        model_6_temp,
    ):
        score_obs_temp = pd.DataFrame(
            [
                [
                    age_temp,
                    tlia_temp,
                    sow_temp,
                    ret_temp,
                    kid_temp,
                    est_temp,
                    well_temp,
                ]
            ],
            columns=["age", "tlia", "sow", "ret", "kid", "est", "well"],
        )
        y1_score_temp = int(
            round(model_1_temp.predict_proba(score_obs_temp)[0, 1] * 100, 0)
        )
        y2_score_temp = int(
            round(model_2_temp.predict_proba(score_obs_temp)[0, 1] * 100, 0)
        )
        y3_score_temp = int(
            round(model_3_temp.predict_proba(score_obs_temp)[0, 1] * 100, 0)
        )
        y4_score_temp = int(
            round(model_4_temp.predict_proba(score_obs_temp)[0, 1] * 100, 0)
        )
        y5_score_temp = int(
            round(model_5_temp.predict_proba(score_obs_temp)[0, 1] * 100, 0)
        )
        y6_score_temp = int(
            round(model_6_temp.predict_proba(score_obs_temp)[0, 1] * 100, 0)
        )
        return (
            y1_score_temp,
            y2_score_temp,
            y3_score_temp,
            y4_score_temp,
            y5_score_temp,
            y6_score_temp,
        )

    # call scoring function
    y1_score, y2_score, y3_score, y4_score, y5_score, y6_score = score_func(
        age,
        tlia,
        sow,
        ret,
        kid,
        est,
        well,
        model1,
        model2,
        model3,
        model4,
        model5,
        model6,
    )

    # define a function to sort goals
    def rank_goals_func(
        y1_score_temp,
        y2_score_temp,
        y3_score_temp,
        y4_score_temp,
        y5_score_temp,
        y6_score_temp,
        xref_table_temp,
    ):
        goals_temp = {
            "Retirement Goal": y1_score_temp,
            "Education Goal": y2_score_temp,
            "Emergency Cash Goal": y3_score_temp,
            "Estate Plan Goal": y4_score_temp,
            "Term Life Insurance Goal": y5_score_temp,
            "Long Term Care Goal": y6_score_temp,
        }
        sorted_goals_temp = sorted(goals_temp.items(), key=lambda x: x[1], reverse=True)
        selected_goals_temp = pd.DataFrame(
            sorted_goals_temp[0:3], columns=["Goals", "Scores"]
        )
        final_table_temp = selected_goals_temp.merge(
            xref_table_temp, how="left", left_on="Goals", right_on="Goals"
        )
        final_table_temp = final_table_temp[
            ["Goals", "EQ Questions", "Resources", "Field Helpfulness"]
        ]
        ## todo: csv
        return final_table_temp

    # call the goals sorter function
    selected_goals = rank_goals_func(
        y1_score,
        y2_score,
        y3_score,
        y4_score,
        y5_score,
        y6_score,
        xref_table,
    )

    return (
        f"""
        Selected Goals
        """,
        generate_table(selected_goals),
    )


if __name__ == "__main__":
    app.run_server(debug=True)
