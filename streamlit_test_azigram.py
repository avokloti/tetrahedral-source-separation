import streamlit as st
from matplotlib import pyplot as plt
import inspect
import textwrap
import numpy as np
from typing import Any

# import some more useful packages!
import librosa # for audio processing
from scipy import signal as scipysig # for filtering

# ---- filtering function ----
def butter_highpass_filter(data, cut, fs, order=5):
    cutoff = 2 * cut / fs
    b, a = scipysig.butter(order, cutoff, btype='high', analog=False)
    y = scipysig.filtfilt(b, a, data)
    return y

# ---- function to convert linear units to dB ----
def logTransform(spec, scale=10**(-5)):
    return 20 * np.log10(spec + scale * np.ones(spec.shape))

# have a function to load in a file and run azigram calculation
# then show azigram masked to certain angles

#@st.cache_data
#def loadData(start_time, end_time):
#    # ---- read in file ----
#    (s, framerate) = librosa.core.load(filename, sr=None, mono=False)
#
#    # ---- get part of signal ----
#    s = s/np.max(s)
#    s = s[:, (start_time * framerate):(end_time * framerate)]
#
#    # ---- filter signal ----
#    s = butter_highpass_filter(s, highpass_filter, framerate)
#
#    # ---- convert to b-format through matrix multiplication ----
#    b_transform = np.asarray([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
#    s_B = b_transform @ s
#
#    # ---- make b-format spectrogram ----
#    specs = []
#    for num in np.arange(4):
#        freqs, inds, spec = scipysig.stft(s_B[num,:], fs=framerate, nperseg=n_fft)
#        nf_full = len(freqs)
#        freqs = freqs[0:nf]
#        specs.append(spec[0:nf, :].T)
#
#    # directly get the three components
#    w = specs[0]
#    x = specs[1]
#    y = specs[2]
#
#    # azimuth values for all pixels
#    azimuth = np.arctan2(np.real(w.conj() * y), np.real(w.conj() * x))
#
#    # weight the azimuth values by the pressure
#    weights = np.abs(w)
#
#    # get grids for time and frequency
#    f_grid, time_grid = np.meshgrid(freqs, inds)
#
#    # need to set these parameters for histogram
#    time_step = 0.1
#    num_time = int(analysis_length * 1/time_step)
#    num_azim = 60
#
#    # get the midpoints of the edges
#    #azims = (azim_edges[0:-1] + azim_edges[1:])/2
#    #time_points = (time_edges[0:-1] + time_edges[1:])/2
#
#    # histogram
#    hist, azim_edges, time_edges = np.histogram2d(x = azimuth.ravel(), y = time_grid.ravel(),
#                                                  bins=[num_azim, num_time],
#                                                  weights = weights.ravel())
#
#    log_hist = np.log(hist + 0.01 * np.ones(hist.shape))
#
#    return azimuth, specs[0], inds, freqs, log_hist, azim_edges, time_edges, framerate, nf_full


@st.cache_data
def loadData(uploaded_file, start_time, end_time):
    # ---- read in file ----
    (s, framerate) = librosa.core.load(uploaded_file, sr=None, mono=False)
    
    # ---- get part of signal ----
    s = s/np.max(s)
    s = s[:, (start_time * framerate):(end_time * framerate)]
    
    # ---- filter signal ----
    s = butter_highpass_filter(s, highpass_filter, framerate)
    
    # ---- convert to b-format through matrix multiplication ----
    b_transform = np.asarray([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
    s_B = b_transform @ s
    return s_B, framerate


# finally, we can apply an inverse FFT to see what the above spectrogram sounds like!
def makeAudioReconstruction(spec, nf_full, framerate, highpass_filter):
    (nx, nf) = spec.shape
    spec_full = np.zeros((nx, nf_full), dtype=np.complex_)
    spec_full[:, 0:nf] = spec
    t, audio = scipysig.istft(spec_full.T, framerate)
    audio = butter_highpass_filter(audio, highpass_filter, framerate)
    audio = audio.astype(np.float32)
    return audio


def animation_demo():
    # Interactive Streamlit elements, like these sliders, return their value.
    # This gives you an extremely simple interaction model.
    st.write("""This app shows how to estimate **acoustic direction-of-arrival** by using a tetrahedral co-located microphone array! First, visualize a 10-second segment of the signal:""")
    
    uploaded_file = st.file_uploader("Choose a .WAV file:", accept_multiple_files=False)
    
    if uploaded_file is not None:
        print('-------')
        print(uploaded_file)
        print('-------')
        
        start_time = st.slider("Start Time", 0, 5*60 - analysis_length, 0)
        end_time = start_time + 10
        
        s_B, framerate = loadData(uploaded_file, start_time, end_time)
        
        # ---- make b-format spectrogram ----
        specs = []
        for num in np.arange(4):
            freqs, inds, spec = scipysig.stft(s_B[num,:], fs=framerate, nperseg=n_fft)
            nf_full = len(freqs)
            freqs = freqs[0:nf]
            specs.append(spec[0:nf, :].T)
    
        print('-------')
        print(spec.shape)
        print('-------')
        
        # directly get the three components
        w = specs[0]
        x = specs[1]
        y = specs[2]

        # azimuth values for all pixels
        azimuth = np.arctan2(np.real(w.conj() * y), np.real(w.conj() * x))

        # weight the azimuth values by the pressure
        weights = np.abs(w)
        
        # get grids for time and frequency
        f_grid, time_grid = np.meshgrid(freqs, inds)
        
        # need to set these parameters for histogram
        time_step = 0.05
        num_time = int(analysis_length * 1/time_step)
        num_azim = 60
        
        # get the midpoints of the edges
        #azims = (azim_edges[0:-1] + azim_edges[1:])/2
        #time_points = (time_edges[0:-1] + time_edges[1:])/2
        
        # histogram
        hist, azim_edges, time_edges = np.histogram2d(x = azimuth.ravel(), y = time_grid.ravel(),
                                                      bins=[num_azim, num_time],
                                                      weights = weights.ravel())
        
        log_hist = np.log(hist + 0.01 * np.ones(hist.shape))
        
        #azimuth, spec, inds, freqs, log_hist, azim_edges, time_edges, framerate, nf_full = loadData(start_time, end_time)
        
        original_audio = makeAudioReconstruction(w, nf_full, framerate, highpass_filter)
        st.audio(original_audio, sample_rate=framerate)
        
        # visualize the resulting masked spectrogram
        fig_original, ax_original = plt.subplots(1, 1, figsize = (10, 4))
        ax_original.pcolormesh(inds, freqs, logTransform(np.abs(w).T))
        ax_original.set_xlabel('Time (sec)', fontsize=14)
        ax_original.set_ylabel('Frequency (Hz)', fontsize=14)
        ax_original.set_title('Original Spectrogram', fontsize=16)
        #plt.colorbar()
        fig_original.tight_layout()
        
        st.pyplot(fig_original)
        
        st.write("""Next, move the sliders on the minimum and maximum azimuth thresholds:""")
        
        # Interactive Streamlit elements, like these sliders, return their value.
        # This gives you an extremely simple interaction model.
        azimuth_lower_limit = st.slider("Minimum Azimuth", -180, 180, -180)
        azimuth_upper_limit = st.slider("Maximum Azimuth", -180, 180, 180)
        
        # mask the spectrogram to only show pixels with azimuth angles in this range
        mask = np.logical_and(180/np.pi * azimuth > azimuth_lower_limit, 180/np.pi * azimuth < azimuth_upper_limit)
        spec_masked = w.copy()
        spec_masked[~mask] = 0
        
        mask = np.logical_and(180/np.pi * azim_edges[0:-1] > azimuth_lower_limit, 180/np.pi * azim_edges[0:-1] < azimuth_upper_limit)
        masked_log_hist = log_hist.copy()
        
        fig_hist, ax_hist = plt.subplots(1, 1, figsize = (10, 4))
        ax_hist.pcolormesh(time_edges[0:-1], 180/np.pi * azim_edges[0:-1], log_hist, cmap='Purples', alpha=0.2)

        ax_hist.pcolormesh(time_edges[0:-1], 180/np.pi * azim_edges[0:-1][mask], masked_log_hist[mask, :], cmap='Purples')
        
        ax_hist.plot([0, analysis_length], [azimuth_lower_limit, azimuth_lower_limit], 'r--')
        ax_hist.plot([0, analysis_length], [azimuth_upper_limit, azimuth_upper_limit], 'r--')
        ax_hist.set_xlabel('Time (sec)', fontsize=14)
        ax_hist.set_ylabel('Azimuth (degrees)', fontsize=14)
        ax_hist.set_xlim([0, analysis_length])
        ax_hist.set_ylim([-180, 180])
        ax_hist.set_title('Histogram over Azimuth and Time', fontsize=16)
        fig_hist.tight_layout()
        
        st.pyplot(fig_hist)
        
        source_audio = makeAudioReconstruction(spec_masked, nf_full, framerate, highpass_filter)
        st.audio(source_audio, sample_rate=framerate)
        
        fig_masked, ax_masked = plt.subplots(1, 1, figsize = (10, 4))
        ax_masked.pcolormesh(inds, freqs, logTransform(np.abs(spec_masked).T))
        ax_masked.set_xlabel('Time (sec)', fontsize=14)
        ax_masked.set_ylabel('Frequency (Hz)', fontsize=14)
        ax_masked.set_title('Masked Spectrogram', fontsize=16)
        #plt.colorbar()
        fig_masked.tight_layout()
        
        # Update the image placeholder by calling the image() function on it.
        st.pyplot(fig_masked)
        
        # Streamlit widgets automatically run the script from top to bottom. Since
        # this button is not connected to any other logic, it just causes a plain
        # rerun.
        st.button("Re-run")


st.set_page_config(page_title="Azigram Thresholding  Demo", page_icon="🦉")
st.markdown("# Azigram Thresholding Demo")
#st.sidebar.header("Azigram Thresholding Demo")

filename = '05_28_21_colorado_refuge_first5min.wav'

analysis_length = 10
n_fft = 1024
nf = 160
highpass_filter = 500

#azimuth, spec, inds, freqs, log_hist, azim_edges, time_edges = loadData()

animation_demo()
