import streamlit as st
import pandas as pd
import pandas as ps
import numpy as np
import plotly_express as px
import plotly.graph_objects as go

from numpy import tile, dot, newaxis, sinc

st.set_page_config(layout="wide")

col1, col2 = st.columns([10, 2])
units = {
    "Hz (hertz)": 1,
    "kHz (kilohertz)": 1000,

}
signals = []
Fig = go.Figure()

if "frequencies" not in st.session_state:
    st.session_state["frequencies"] = []

if "amplitudes" not in st.session_state:
    st.session_state["amplitudes"] = []
if "time" not in st.session_state:
    st.session_state["time"] = np.arange(0, 10, 0.0001)

if "signal" not in st.session_state:
    st.session_state["signal"] = np.zeros(len(st.session_state["time"]))


def add_signal_plotting(amp):
    for i in range(len(st.session_state["amplitudes"])):
        if i == 0:
            st.session_state["time"] = np.arange(0, 10, 0.0001)
            st.session_state["signal"] = np.zeros(
                len(st.session_state["time"]))

        st.session_state["signal"] += st.session_state["amplitudes"][i] * np.sin(
            2 * np.pi * st.session_state["frequencies"][i] * st.session_state["time"])

    if len(st.session_state["amplitudes"]) != 0:

        fig = px.line(x=st.session_state["time"].tolist(), y=st.session_state["signal"].tolist(
        ), labels={'x': 'Time (seconds)', 'y': 'Amplitude'})
        Fig.add_trace(go.Scatter(
            x=st.session_state["time"].tolist(), y=st.session_state["signal"].tolist(), mode='lines', line=go.scatter.Line(color="dark blue"), name="generated signal"))
    else:

        st.session_state["time"] = np.arange(0, 10, 0.0001)
        st.session_state["signal"] = np.zeros(len(st.session_state["time"]))
        fig = px.line(x=st.session_state["time"].tolist(), y=st.session_state["signal"].tolist(
        ), labels={'x': 'Time (seconds)', 'y': 'Amplitude'})

        Fig.add_trace(go.Scatter(
            x=st.session_state["time"].tolist(), y=st.session_state["signal"].tolist(), mode='lines', line=go.scatter.Line(color="dark blue"), name="generated signal"))

    st.plotly_chart(fig, use_container_width=True)

    return amp


def signal_adder(frequency, amplitude):
    st.session_state["frequencies"].append(frequency)
    st.session_state["amplitudes"].append(amplitude)


def signal_deleter(selected_signal):
    index = signals.index(selected_signal)
    del st.session_state["frequencies"][index]
    del st.session_state["amplitudes"][index]
    st.experimental_rerun()


def sample(time, amplitude, fs):
    if len(time) == len(amplitude):
        step = time[len(time)-1]-time[0]
        points_per_indices = int((len(time) / step) / fs)
        amplitude = amplitude[::points_per_indices]
        time = time[::points_per_indices]
    return time, amplitude


def sinc_interp(sampled_Amp, sampled_Time, Time):

    if len(sampled_Amp) != len(sampled_Time):
        raise Exception(
            'sampled_Amp and sampled_Time must be the same length')

    # Find the period
    T = sampled_Time[1] - sampled_Time[0]

    sincM = tile(Time, (len(sampled_Time), 1)) - \
        tile(sampled_Time[:, newaxis], (1, len(Time)))
    y = dot(sampled_Amp, sinc(sincM / T))
    return y


def Plotting(time, signal, samplefreq, recon_time, recon_signal):

    Fig.add_trace(go.Scatter(
        x=time, y=signal, mode='lines', line=go.scatter.Line(color="black"), name="signal"))
    Fig.update_xaxes(title_text="Time (s)")
    Fig.update_yaxes(title_text="Amplitude (mV)")
    sampled_Time, sampled_Amp = sample(time, signal, samplefreq)
    Fig.add_trace(go.Scatter(x=sampled_Time, y=sampled_Amp,
                  mode='markers', name='Sampling'))

    Fig.add_trace(go.Scatter(
        x=recon_time, y=recon_signal, mode='lines', name='Reconstructed Signal', line=go.scatter.Line(color="grey")))
    st.plotly_chart(Fig, use_container_width=True)


def PlotRecons(time, Signal):
    Fig = go.Figure()
    Fig.add_trace(go.Scatter(
        x=time, y=Signal, mode='lines', name='Reconstructed Signal'))
    Fig.update_xaxes(title_text="Time (s)")
    Fig.update_yaxes(title_text="Amplitude (mV)")

    st.plotly_chart(Fig, use_container_width=True)


with col2:

    frequency_units = st.selectbox('Select Frequency units', options=units)

    frequency = st.slider(min_value=1.0, max_value=20.0, step=0.1,
                          label="Frequency")*units[frequency_units]

    amplitude = st.slider(min_value=1.0, max_value=20.0,
                          step=0.1, label="Amplitude")

    if st.button(label="Add Signal"):
        signal_adder(frequency, amplitude)

    selected_signal = st.selectbox(label="Choose signal", options=signals)

    if st.button(label="Delete Signal"):
        if len(st.session_state["amplitudes"]) != 0:
            signal_deleter(selected_signal)

    signal_csv = pd.DataFrame({"Time": st.session_state["time"].tolist(
    ), "Value": st.session_state["signal"].tolist()}).to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download the signal as CSV",
        data=signal_csv,
        file_name='generated_signal.csv',
        mime='text/csv',
    )

uploaded_file = st.sidebar.file_uploader("Upload your file here", type="CSV")


if uploaded_file is not None:
    fs = st.sidebar.slider('Sample Frequency', 1, 200, 25)
    noise = st.sidebar.checkbox('Add noise')

    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df
    df = st.session_state['df']
    time = df.iloc[:, 0].to_numpy()
    amp = df.iloc[:, 1].to_numpy()
    if (noise):
        SNR = st.sidebar.slider('SNR (dBw)', 0.01, 100.0,
                                20.0, step=0.5)

        # Calculate signal power and convert to dB
        x_watts = amp ** 2
        sig_avg_watts = np.mean(x_watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)

        # Calculate noise according to [2] then convert to watts
        noise_avg_db = sig_avg_db - SNR
        noise_avg_watts = 10 ** (noise_avg_db / 10)

        # Generate an sample of white noise
        mean_noise = 0
        noise_volts = np.random.normal(
            mean_noise, np.sqrt(noise_avg_watts), len(x_watts))

        # Noise up the original signal
        noisedSignal = amp + noise_volts

        with col1:
            add_signal_plotting()
            sampled_Time, sampled_Amp = sample(time, noisedSignal, fs)
            Plotting(time, noisedSignal, fs, time, sinc_interp(
                sampled_Amp, sampled_Time, time))

    else:
        with col1:
            add_signal_plotting()
            sampled_Time, sampled_Amp = sample(time, amp, fs)
            Plotting(time, amp, fs, time, sinc_interp(
                sampled_Amp, sampled_Time, time))


else:
    default_file = ps.read_csv('samples/sine_wave.csv')
    fs = st.sidebar.slider('Sample Frequency', 1, 200, 25)
    noise = st.sidebar.checkbox('Add noise')

    df = default_file
    st.session_state['df'] = df
    df = st.session_state['df']
    time = df.iloc[:, 0].to_numpy()
    amp = df.iloc[:, 1].to_numpy()
    if (noise):
        SNR = st.sidebar.slider('SNR (dBw)', 0.01, 100.0,
                                20.0, step=0.5)

        # Calculate signal power and convert to dB
        x_watts = amp ** 2
        sig_avg_watts = np.mean(x_watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        # Calculate noise according to [2] then convert to watts
        noise_avg_db = sig_avg_db - SNR
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise_volts = np.random.normal(
            mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
        # Noise up the original signal
        noisedSignal = amp + noise_volts
        sampled_Time, sampled_Amp = sample(time, noisedSignal, fs)
        with col1:
            add_signal_plotting()
            Plotting(time, noisedSignal, fs, time, sinc_interp(
                sampled_Amp, sampled_Time, time))

            # PlotRecons(time, sinc_interp(sampled_Amp, sampled_Time, time))

    else:
        with col1:

            amp = add_signal_plotting(amp)
            # st.header(amp)
            sampled_Time, sampled_Amp = sample(time, amp, fs)
            Plotting(time, amp, fs, time, sinc_interp(
                sampled_Amp, sampled_Time, time))

            # PlotRecons(time, sinc_interp(sampled_Amp, sampled_Time, time))
