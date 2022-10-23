import streamlit as st
import plotly_express as px
import numpy as np

   
if "frequencies" not in st.session_state:
    st.session_state["frequencies"] = []

if "amplitudes" not in st.session_state:
    st.session_state["amplitudes"] = []

if "signal_amplitude" not in st.session_state:
    st.session_state["signal_amplitude"] = 0

if "signal_frequency" not in st.session_state:
    st.session_state["signal_frequency"] = 0


def signal_adder (frequency, amplitude):
    st.session_state["frequencies"].append(frequency) 
    st.session_state["amplitudes"].append(amplitude)

def signal_deleter (selected_signal):
    index = signals.index(selected_signal)
    del st.session_state["frequencies"][index]
    del st.session_state["amplitudes"][index]
    
units = {
    "Hz (hertz)": 1,
    "kHz (kilohertz)": 1000,
    
}




frequency_units=st.selectbox('Select Frequency units', options=units)

frequency=st.slider(min_value= 1, max_value=20, label="Frequency")*units[frequency_units]

amplitude=st.slider(min_value= 1, max_value=20, label="Amplitude")

if st.button(label="Add Signal"):
    signal_adder(frequency,amplitude)
    

st.markdown("## Preview")
wave_time = np.arange(0, np.pi,0.0001)

preview_fig = px.line(x=wave_time, y=np.sin(2*np.pi*wave_time*frequency)*amplitude, labels={'x':'Time (seconds)', 'y':'Amplitude'})
st.plotly_chart(preview_fig)



st.markdown("## Wave")


st.session_state["signal_amplitude"] = 0
for x in st.session_state["amplitudes"]:

    st.session_state["signal_amplitude"] +=  x

st.session_state["signal_frequency"] = 0
for x in st.session_state["frequencies"]:

    st.session_state["signal_frequency"] +=  x


fig = px.line(x=wave_time, y=np.sin(2*np.pi*wave_time*st.session_state["signal_frequency"])*st.session_state["signal_amplitude"], labels={'x':'Time (seconds)', 'y':'Amplitude'})
st.plotly_chart(fig)


st.markdown("### Signals")
signals = []

for x in range(len(st.session_state["amplitudes"])):
    signals.append(f"Signal {x+1}:  Frequency: {st.session_state['frequencies'][x]} Amplitude: {st.session_state['amplitudes'][x]} ")


selected_signal = st.selectbox(label="Choose signal", options=signals)

if st.button(label="Delete Signal"):
    signal_deleter(selected_signal)
    



### Uncomment the following block for debugging:


# st.markdown("Amp")
# st.write(st.session_state["amplitudes"])
# st.markdown("Freq")

# st.write(st.session_state["frequencies"])

# st.markdown("Amp")
# st.write(st.session_state["signal_amplitude"])

# st.markdown("Freq")
# st.write(st.session_state["signal_frequency"])



# st.write(len(st.session_state["amplitudes"]))
