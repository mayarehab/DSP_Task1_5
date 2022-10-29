import streamlit as st
import plotly_express as px
import numpy as np
import pandas as pd


def load_view():

    if "frequencies" not in st.session_state:
        st.session_state["frequencies"] = []

    if "amplitudes" not in st.session_state:
        st.session_state["amplitudes"] = []

   
        
    # fixing signal generator
        
    if "time" not in st.session_state:
        st.session_state["time"] = np.arange(0, np.pi, 0.0001)

    if "signal" not in st.session_state:
        st.session_state["signal"] = np.zeros(len(st.session_state["time"]))
        

    def signal_adder(frequency, amplitude):
        st.session_state["frequencies"].append(frequency)
        st.session_state["amplitudes"].append(amplitude)

    def signal_deleter(selected_signal):
        index = signals.index(selected_signal)
        del st.session_state["frequencies"][index]
        del st.session_state["amplitudes"][index]
        st.experimental_rerun()

    units = {
        "Hz (hertz)": 1,
        "kHz (kilohertz)": 1000,

    }

    frequency_units = st.selectbox('Select Frequency units', options=units)

    frequency = st.slider(min_value=1.0, max_value=20.0, step=0.1,
                          label="Frequency")*units[frequency_units]

    amplitude = st.slider(min_value=1.0, max_value=20.0,
                          step=0.1, label="Amplitude")

    if st.button(label="Add Signal"):
        signal_adder(frequency, amplitude)

    st.markdown("## Preview")
    wave_time = np.arange(0, np.pi, 0.0001)

    preview_fig = px.line(x=wave_time, y=np.sin(2*np.pi*wave_time*frequency)
                          * amplitude, labels={'x': 'Time (seconds)', 'y': 'Amplitude'})
    st.plotly_chart(preview_fig, use_container_width=True)

    st.markdown("## Wave")

    # fixing signal generator


    for i in range(len(st.session_state["amplitudes"])):
        if i == 0:
            st.session_state["time"] = np.arange(0, np.pi, 0.0001)
            st.session_state["signal"] = np.zeros(len(st.session_state["time"]))
            
        st.session_state["signal"] += st.session_state["amplitudes"][i] * np.sin(2 * np.pi * st.session_state["frequencies"][i] * st.session_state["time"])


   

    if len(st.session_state["amplitudes"]) != 0:
        fig = px.line(x=st.session_state["time"].tolist(), y=st.session_state["signal"].tolist(), labels={'x': 'Time (seconds)', 'y': 'Amplitude'})
    else:
        st.session_state["time"] = np.arange(0, np.pi, 0.0001)
        st.session_state["signal"] = np.zeros(len(st.session_state["time"]))
        fig = px.line(x=st.session_state["time"].tolist(), y=st.session_state["signal"].tolist(), labels={'x': 'Time (seconds)', 'y': 'Amplitude'})


    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Signals")
    signals = []

    for x in range(len(st.session_state["amplitudes"])):
        signals.append(
            f"Signal {x+1}:  Frequency: {st.session_state['frequencies'][x]} Amplitude: {st.session_state['amplitudes'][x]} ")

    selected_signal = st.selectbox(label="Choose signal", options=signals)

    if st.button(label="Delete Signal"):
        if len(st.session_state["amplitudes"]) != 0:
            signal_deleter(selected_signal)

    signal_csv = pd.DataFrame({"Time": st.session_state["time"].tolist(), "Value": st.session_state["signal"].tolist()}).to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download the signal as CSV",
        data=signal_csv,
        file_name='generated_signal.csv',
        mime='text/csv',
    )
