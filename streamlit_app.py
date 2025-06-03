#SUKSESSSSSSSSSSSSSSSSSSS

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import soundfile as sf
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Judul dan header aplikasi Streamlit
st.title("Sinyal dan Sistem Biomedik")
st.header("**Group 2 Final Project (Band-Pass Filter Design and Audio Processing)**")

# Sidebar untuk mengatur parameter filter
st.sidebar.title("Parameter Filter")

transition_width = st.sidebar.slider("Transition Width (Hz)", 10, 5000, 500, key="transition_width")
stopband_attenuation = st.sidebar.slider("Stopband Attenuation (dB)", 20, 100, 60, key="stopband_attenuation")
fs = st.sidebar.slider("Sampling Frequency (Hz)", 8000, 48000, 44100, step=1000, key="sampling_freq")
lowcut = st.sidebar.slider("Low Cutoff Frequency (Hz)", 20, fs//2 - 1000, 1000, key="lowcut")
highcut = st.sidebar.slider("High Cutoff Frequency (Hz)", lowcut + 100, fs//2, 5000, key="highcut")
amplitude_max = st.sidebar.slider("Perbesaran Amplitude (Max Magnitude in dB)", -300, 10, 0, key="amplitude_max")
amplitude_min = st.sidebar.slider("Perbesaran Amplitude (Min Magnitude in dB)", -300, 10, -300, key="amplitude_min")

fc = np.sqrt(lowcut * highcut)
st.sidebar.markdown(f"**Frekuensi Tengah (fc)**: {fc:.2f} Hz")

nyq = 0.5 * fs
normalized_transition_width = transition_width / nyq
st.sidebar.markdown(f"**Lebar Transisi (Δf, dinormalisasi)**: {normalized_transition_width:.4f}")

if lowcut >= highcut:
    st.sidebar.error("Low cutoff harus lebih rendah daripada High Cutoff.")
elif highcut >= fs / 2:
    st.sidebar.error("High cutoff harus lebih rendah dari setengah Frequency Sample (fs/2).")
elif amplitude_min >= amplitude_max:
    st.sidebar.error("Min Magnitude harus lebih kecil dari Max Magnitude.")

window_options = ["Rectangular", "Hamming", "Hanning", "Blackman", "Kaiser", "All Window"]
window_type = st.sidebar.selectbox("Jenis Window yang ingin digunakan", window_options)

# Fungsi untuk mendesain band-pass filter FIR dengan window tertentu
def design_bandpass_filter(fs, lowcut, highcut, transition_width, stopband_attenuation, window_type):
    nyq = 0.5 * fs
    delta_f = transition_width / nyq  # Normalisasi lebar transisi
    if window_type == "Kaiser":
        A = stopband_attenuation
        N = (A - 8) / (2.285 * 2 * np.pi * delta_f)
        N = int(np.ceil(N))
        if N < 3:  # Minimal N untuk filter yang masuk akal
            N = 3
        if N % 2 == 0:
            N += 1
        if A > 50:
            beta = 0.1102 * (A - 8.7)
        elif A >= 21:
            beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21)
        else:
            beta = 0.0
        window = ("kaiser", beta)
    elif window_type == "Rectangular":
        N = 0.9 / delta_f
        N = int(np.ceil(N))
        if N < 3:
            N = 3
        if N % 2 == 0:
            N += 1
        window = "boxcar"
    elif window_type == "Hanning":
        N = 3.1 / delta_f
        N = int(np.ceil(N))
        if N < 3:
            N = 3
        if N % 2 == 0:
            N += 1
        window = "hann"
    elif window_type == "Hamming":
        N = 3.3 / delta_f
        N = int(np.ceil(N))
        if N < 3:
            N = 3
        if N % 2 == 0:
            N += 1
        window = "hamming"
    elif window_type == "Blackman":
        N = 5.5 / delta_f
        N = int(np.ceil(N))
        if N < 3:
            N = 3
        if N % 2 == 0:
            N += 1
        window = "blackman"
    else:
        N, beta = signal.kaiserord(stopband_attenuation, delta_f)
        if N % 2 == 0:
            N += 1
        window = window_type
    taps = signal.firwin(N, [lowcut / nyq, highcut / nyq], window=window, pass_zero=False)  # Normalisasi cutoff
    return taps, N

st.subheader("Upload File Audio (.wav / .mp3)")
uploaded_file = st.file_uploader("Upload file", type=['wav', 'mp3'])

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    if file_ext not in ['.wav', '.mp3']:
        st.error("Format file tidak sesuai! Harap upload file .wav atau .mp3.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=None)
        except Exception as e:
            st.error(f"Gagal memroses file audio: {str(e)}")
            st.stop()

        st.write(f"Sample rate: {sr} Hz")
        st.write(f"Duration: {len(y)/sr:.2f} seconds")

        if sr != fs:
            st.warning(f"Resampling dari {sr} Hz ke {fs} Hz...")
            y = librosa.resample(y, orig_sr=sr, target_sr=fs)

        if window_type == "All Window":
            filter_taps, _ = design_bandpass_filter(fs, lowcut, highcut, transition_width, stopband_attenuation, "Kaiser")
            y_filtered = signal.lfilter(filter_taps, 1.0, y)

            st.subheader("Grafik perbandingan Original Audio vs Filtered Audio")
            fig_time, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            t = np.linspace(0, len(y)/fs, len(y))
            axs[0].plot(t, y, color='#1E90FF')  # Dodger blue for original
            axs[0].set_title("Original Audio", color='white')
            axs[0].set_ylabel("Amplitude", color='white')
            axs[1].plot(t, y_filtered, color='#00008B')  # Dark blue for filtered
            axs[1].set_title("Filtered Audio (Rectangular Window)", color='white')
            axs[1].set_xlabel("Time (s)", color='white')
            axs[1].set_ylabel("Amplitude", color='white')
            for ax in axs:
                ax.set_facecolor('none')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.tick_params(colors='white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
            fig_time.patch.set_alpha(0.0)  # Background transparan
            st.pyplot(fig_time)

            st.subheader("Grafik Perbandingan Spektrum Frekuensi (FFT)")
            fig_fft, axs_fft = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            freqs = np.fft.rfftfreq(len(y), 1/fs)
            fft_original = np.abs(np.fft.rfft(y)) / len(y)
            fft_filtered = np.abs(np.fft.rfft(y_filtered)) / len(y_filtered)
            fft_original_db = 20 * np.log10(fft_original + 1e-10)  # Convert to dB
            fft_filtered_db = 20 * np.log10(fft_filtered + 1e-10)
            axs_fft[0].plot(freqs, fft_original_db, color='#1E90FF')
            axs_fft[0].set_title("FFT Original Audio", color='white')
            axs_fft[0].set_ylabel("Magnitude (dB)", color='white')
            axs_fft[1].plot(freqs, fft_filtered_db, color='#00008B')
            axs_fft[1].set_title("FFT Filtered Audio (Rectangular Window)", color='white')
            axs_fft[1].set_xlabel("Frequency (Hz)", color='white')
            axs_fft[1].set_ylabel("Magnitude (dB)", color='white')
            for ax in axs_fft:
                ax.set_facecolor('none')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.tick_params(colors='white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
            fig_fft.patch.set_alpha(0.0)  # Background transparan
            st.pyplot(fig_fft)

            st.subheader("Frequency Response dari Filter")
            st.markdown(f"Filter didesain untuk **{window_type}**")

            fig_resp, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            window_options_without_all = ["Rectangular", "Hamming", "Hanning", "Blackman", "Kaiser"]
            for win in window_options_without_all:
                taps, _ = design_bandpass_filter(fs, lowcut, highcut, transition_width, stopband_attenuation, win)
                w, h = signal.freqz(taps, worN=8000, fs=fs)
                magnitude = 20 * np.log10(np.abs(h))
                phase = np.unwrap(np.angle(h))
                ax1.plot(w, magnitude, label=win)
                ax2.plot(w, phase, label=win)

            ax1.set_title("Magnitude Response tiap Window", color='white')
            ax1.set_ylabel("Magnitude (dB)", color='white')
            ax1.axvline(lowcut, color='gray', linestyle='--', linewidth=0.8)
            ax1.axvline(highcut, color='gray', linestyle='--', linewidth=0.8)
            ax1.grid()
            ax1.legend()
            ax2.set_title("Phase Response tiap Window", color='white')
            ax2.set_xlabel("Phase Response (rad)", color='white')
            ax2.set_ylabel("Phase (rad)", color='white')
            ax2.grid()
            ax2.legend()
            for ax in [ax1, ax2]:
                ax.set_facecolor('none')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.tick_params(colors='white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
            fig_resp.patch.set_alpha(0.0)  # Background transparan
            st.pyplot(fig_resp)
            st.stop()

        filter_taps, N = design_bandpass_filter(fs, lowcut, highcut, transition_width, stopband_attenuation, window_type)
        order = N - 1  # Order = number of coefficients - 1
        st.sidebar.markdown(f"*Filter Order*: {order}")

        # Menampilkan koefisien filter dalam bentuk tabel dengan kolom Index dan Value
        st.subheader("Koefisien Filter (h[n])")
        st.write(f"Jumlah Koefisien Filter: {len(filter_taps)}")
        filter_coeffs = {
            "Index": np.arange(len(filter_taps)),
            "Value": filter_taps
        }
        filter_coeffs_df = pd.DataFrame(filter_coeffs)
        st.dataframe(filter_coeffs_df, height=200)

        # Download button untuk jumlah koefisien
        coeff_count = pd.DataFrame({"Number of Coefficients": [len(filter_taps)]})
        csv = coeff_count.to_csv(index=False)
        st.download_button(
            label="Download Number of Coefficients",
            data=csv,
            file_name="number_of_coefficients.csv",
            mime="text/csv"
        )

        # Menampilkan parameter lainnya
        st.subheader("Parameter Filter")
        st.write("**Lebar Transisi (Δf)**:", transition_width, "Hz")
        st.write("**Panjang Koefisien Filter (N)**:", N)
        st.write("**Orde Filter**:", order)

        y_filtered = signal.lfilter(filter_taps, 1.0, y)
        filtered_path = tmp_path.replace(".wav", "_filtered.wav").replace(".mp3", "_filtered.wav")
        sf.write(filtered_path, y_filtered, fs)

        st.subheader("Pemutaran Audio")
        col_audio1, col_audio2 = st.columns(2)
        with col_audio1:
            st.write("Original Audio:")
            st.audio(tmp_path, format='audio/wav')
            with open(tmp_path, "rb") as file:
                st.download_button(
                    label="Download Original Audio",
                    data=file,
                    file_name=os.path.basename(tmp_path),
                    mime="audio/wav"
                )
        with col_audio2:
            st.write("Filtered Audio:")
            st.audio(filtered_path, format='audio/wav')
            with open(filtered_path, "rb") as file:
                st.download_button(
                    label="Download Filtered Audio",
                    data=file,
                    file_name=os.path.basename(filtered_path),
                    mime="audio/wav"
                )

        fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        t = np.linspace(0, len(y)/fs, len(y))
        axs[0].plot(t, y, color='#1E90FF')  # Dodger blue for original
        axs[0].set_title("Original Audio", color='white')
        axs[0].set_ylabel("Amplitude", color='white')
        axs[1].plot(t, y_filtered, color='#00008B')  # Dark blue for filtered
        axs[1].set_title("Filtered Audio", color='white')
        axs[1].set_xlabel("Time (s)", color='white')
        axs[1].set_ylabel("Amplitude", color='white')
        for ax in axs:
            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
        fig.patch.set_alpha(0.0)  # Background transparan
        st.subheader("Grafik perbandingan Original Audio vs Filtered Audio")
        st.pyplot(fig)

        st.subheader("Grafik Perbandingan Spektrum Frekuensi (FFT)")
        fig_fft, axs_fft = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        freqs = np.fft.rfftfreq(len(y), 1/fs)
        fft_original = np.abs(np.fft.rfft(y)) / len(y)
        fft_filtered = np.abs(np.fft.rfft(y_filtered)) / len(y_filtered)
        fft_original_db = 20 * np.log10(fft_original + 1e-10)  # Convert to dB
        fft_filtered_db = 20 * np.log10(fft_filtered + 1e-10)
        axs_fft[0].plot(freqs, fft_original_db, color='#1E90FF')
        axs_fft[0].set_title("FFT Original Audio", color='white')
        axs_fft[0].set_ylabel("Magnitude (dB)", color='white')
        axs_fft[1].plot(freqs, fft_filtered_db, color='#00008B')
        axs_fft[1].set_title("FFT Filtered Audio", color='white')
        axs_fft[1].set_xlabel("Frequency (Hz)", color='white')
        axs_fft[1].set_ylabel("Magnitude (dB)", color='white')
        for ax in axs_fft:
            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
        fig_fft.patch.set_alpha(0.0)  # Background transparan
        st.pyplot(fig_fft)

        st.subheader("Frequency Response dari Filter")
        st.markdown(f"Filter didesain menggunakan window: **{window_type}**")
        w, h = signal.freqz(filter_taps, worN=8000, fs=fs)
        amplitude_response = 20 * np.log10(np.abs(h))
        phase_response = np.unwrap(np.angle(h))  # Already in radians

        # Membuat 2 subplot menggunakan Plotly
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Magnitude Response", "Phase Response"),
            vertical_spacing=0.15,  # Increased spacing to avoid overlap
            row_heights=[0.5, 0.5]
        )

        # Grafik 1: Magnitude Response
        fig.add_trace(
            go.Scatter(x=w, y=amplitude_response, mode='lines', name='Magnitude (dB)'),
            row=1, col=1
        )
        fig.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="-3 dB", row=1, col=1)
        fig.add_vline(x=lowcut, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_vline(x=highcut, line_dash="dash", line_color="gray", row=1, col=1)
        fig.update_yaxes(title_text="Magnitude (dB)", range=[amplitude_min, amplitude_max], row=1, col=1)

        # Grafik 2: Phase Response
        fig.add_trace(
            go.Scatter(x=w, y=phase_response, mode='lines', name='Phase (rad)'),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Phase (rad)", row=2, col=1)
        fig.update_xaxes(title_text="Phase Response (rad)", row=2, col=1)

        # Update layout untuk tampilan yang lebih rapi
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Grafik Spektogram
        st.subheader("Grafik Spectrogram")
        fig_spec, axs_spec = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D_orig, sr=fs, x_axis='time', y_axis='hz', ax=axs_spec[0])
        axs_spec[0].set_title('Original Audio Spectrogram')
        axs_spec[0].set_ylabel('Frequency (Hz)')
        D_filtered = librosa.amplitude_to_db(np.abs(librosa.stft(y_filtered)), ref=np.max)
        librosa.display.specshow(D_filtered, sr=fs, x_axis='time', y_axis='hz', ax=axs_spec[1])
        axs_spec[1].set_title(f'Filtered Audio Spectrogram)')
        axs_spec[1].set_xlabel('Time (s)')
        axs_spec[1].set_ylabel('Frequency (Hz)')
        plt.tight_layout()
        st.pyplot(fig_spec)
        fig_spec.patch.set_alpha(0.0)

else:
    st.info("Silakan upload file audio (.wav / .mp3) untuk diproses.")
