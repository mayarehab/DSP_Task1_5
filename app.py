import streamlit as st
import pandas as pd
import pandas as ps
import numpy as np
from numpy import tile, dot, newaxis, sinc
import plotly.graph_objects as go

st.set_page_config(layout="wide")




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

def Plotting(time, signal, samplefreq, recon_time,recon_signal):
        Fig = go.Figure()
        Fig.add_trace(go.Scatter(
            x=time, y=signal, mode='lines', line=go.scatter.Line(color="black")))
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
            
            sampled_Time, sampled_Amp = sample(time, noisedSignal, fs)
            Plotting(time, noisedSignal, fs,time, sinc_interp(sampled_Amp, sampled_Time, time))
            
            

        else:
           
            sampled_Time, sampled_Amp = sample(time, amp, fs)
            Plotting(time, amp, fs, time, sinc_interp(sampled_Amp, sampled_Time, time))
            
           

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
            noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
            # Noise up the original signal
            noisedSignal = amp + noise_volts
            sampled_Time, sampled_Amp = sample( time,noisedSignal, fs)
            Plotting(time,noisedSignal,fs,time,sinc_interp(sampled_Amp, sampled_Time, time))
            
            PlotRecons (time,sinc_interp(sampled_Amp, sampled_Time, time))

        else:
            sampled_Time, sampled_Amp = sample( time,amp, fs)
            Plotting(time,amp,fs, time,sinc_interp(sampled_Amp, sampled_Time, time))
            
            PlotRecons (time,sinc_interp(sampled_Amp, sampled_Time, time))
