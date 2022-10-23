import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from typing import Dict
from typing import Set
from streamlit_plotly_events import plotly_events

df = st.session_state['df']


def initialize_state():
    for q in ["latitude_to_depth", "date_to_time", "Date"]:
        if f"{q}_query" not in st.session_state:
            st.session_state[f"{q}_query"] = set()

    if "counter" not in st.session_state:
        st.session_state.counter = 0


def reset_state_callback():
    
    st.session_state.counter = 1 + st.session_state.counter

    for q in ["latitude_to_depth", "date_to_time", "Date"]:
        st.session_state[f"{q}_query"] = set()


def query_data(df = st.session_state['df']) -> df:
    df["latitude_to_depth"] = (
        (100 * df["magnitude"]).astype(int).astype(str)
        + "-"
        + (100 * df["tip"]).astype(int).astype(str)
    )
    df["date_to_time"] = df["magnitude"].astype(str) + "-" + df["time"].astype(str)
    df["selected"] = True

    for q in ["latitude_to_depth", "date_to_time", "Date"]:
        if st.session_state[f"{q}_query"]:
            df.loc[~df[q].isin(st.session_state[f"{q}_query"]), "selected"] = False

    return df


def build_latitude_to_depth_figure(df = st.session_state['df']) -> go.Figure:
    fig = px.scatter(
        df,
        "magnitude",
        "tip",
        color="selected",
        color_discrete_sequence=["rgba(99, 110, 250, 0.2)", "rgba(99, 110, 250, 1)"],
        category_orders={"selected": [False, True]},
        hover_data=[
            "magnitude",
            "tip",
            "Date",
        ],
        height=800,
    )
    fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
    fig.update_xaxes(gridwidth=0.1, gridcolor="#EDEDED")
    fig.update_yaxes(gridwidth=0.1, gridcolor="#EDEDED")
    return fig


def build_date_to_time_figure(df = st.session_state['df']) -> go.Figure:
    return px.density_heatmap(df[df["selected"] == True], "magnitude", "time", height=400)


def build_Date_figure(df = st.session_state['df']) -> go.Figure:
    return px.histogram(
        df,
        color="selected",
        color_discrete_sequence=["rgba(99, 110, 250, 1)", "rgba(99, 110, 250, 0.2)"],
        category_orders={
            "selected": [True, False],
        },
        height=400,
    )


def render_preview_ui(df = st.session_state['df']):
    with st.expander("Preview"):
        l, r = st.columns(2)
        l.dataframe(
            df,
        )
        r.json(
            {
                k: v
                for k, v in st.session_state.to_dict().items()
                if f'_{st.session_state["counter"]}' not in k
            }
        )


def render_plotly_ui(transformed_df = st.session_state['df']) -> Dict:
    c1, c2 = st.columns(2)

    latitude_to_depth_figure = build_latitude_to_depth_figure(transformed_df)
    date_to_time_figure = build_date_to_time_figure(transformed_df)
    Date_figure = build_Date_figure(transformed_df)

    with c1:
        latitude_to_depth_selected = plotly_events(
            latitude_to_depth_figure,
            select_event=True,
            key=f"latitude_to_depth_{st.session_state.counter}",
        )
    with c2:
        date_to_time_clicked = plotly_events(
            date_to_time_figure,
            click_event=True,
            key=f"date_to_time_{st.session_state.counter}",
        )
        Date_clicked = plotly_events(
            Date_figure,
            click_event=True,
            key=f"Date_{st.session_state.counter}",
        )

    current_query = {}
    current_query["latitude_to_depth_query"] = {
        f"{int(100*el['x'])}-{int(100*el['y'])}" for el in latitude_to_depth_selected
    }
    current_query["date_to_time_query"] = {
        f"{el['x']}-{el['y']}" for el in date_to_time_clicked
    }
    current_query["Date_query"] = {el["x"] for el in Date_clicked}

    return current_query


def update_state(current_query: Dict[str, Set]):
    rerun = False
    for q in ["latitude_to_depth", "date_to_time", "Date"]:
        if current_query[f"{q}_query"] - st.session_state[f"{q}_query"]:
            st.session_state[f"{q}_query"] = current_query[f"{q}_query"]
            rerun = True

    if rerun:
        st.experimental_rerun()


def main():
    df = st.session_state['df']
    transformed_df = query_data(df)

    st.title("Plotly events")
    render_preview_ui(transformed_df)

    current_query = render_plotly_ui(transformed_df)

    update_state(current_query)

    st.button("Reset filters", on_click=reset_state_callback)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    initialize_state()
    main()