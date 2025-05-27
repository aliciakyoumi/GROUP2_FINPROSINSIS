import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import soundfile as sf
import tempfile
import os

st.title("Sinyal dan Sistem Biomedik")
st.header("**Group 2 Final Project (Band-Pass Filter Design and Audio Processing)**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("User Input")
    user_name = st.text_input("Enter your name:", "Student")
    st.write(f"Hello, {user_name}!")
    age = st.number_input("Enter your age:", min_value=0, max_value=120, value=20)
    st.write(f"You are {age} years old.")

# Sidebar: Parameter filter
st.sidebar.title("Parameter Filter")
transition_width = st.sidebar.slider("Transition Width (Hz)", 10, 500, 50)
stopband_attenuation = st.sidebar.slider("Stopband Attenuation (dB)", 20, 100, 60)
fs = st.sidebar.slider("Sampling Frequency (Hz)", 8000, 48000, 44100, step=1000)
lowcut = st.sidebar.slider("Low Cutoff Frequency (Hz)", 20, fs//2 - 1000, 1000)
highcut = st.sidebar.slider("High Cutoff Frequency (Hz)", lowcut + 100, fs//2, 5000)

if lowcut >= highcut:
    st.sidebar.error("Low cutoff harus lebih rendah daripada High Cutoff.")
elif highcut >= fs / 2:
    st.sidebar.error("High cutoff harus lebih rendah dari setengah Frequency Sample (fs/2).")


window_options = ["Kaiser", "Hamming", "Hanning", "Blackman", "Rectangular"]
window_type = st.sidebar.selectbox("Jenis Window yang ingin digunakan", window_options + ["Tampilan Tiap Window"])

# Fungsi desain filter
def design_bandpass_filter(fs, lowcut, highcut, transition_width, stopband_attenuation, window_type):
    nyq = 0.5 * fs
    width = transition_width / nyq
    N, beta = signal.kaiserord(stopband_attenuation, width)
    if N % 2 == 0:
        N += 1

    if window_type == "Kaiser":
        window = ("kaiser", beta)
    elif window_type == "Rectangular":
        window = "boxcar"
    elif window_type == "Hanning":
        window = "hann"
    elif window_type == "Hamming":
        window = "hamming"
    elif window_type == "Blackman":
        window = "blackman"
    else:
        window = window_type

    taps = signal.firwin(N, [lowcut, highcut], window=window, pass_zero=False, fs=fs)
    return taps, N

# File uploader
st.subheader("Upload File Audio (.wav / .mp3)")
uploaded_file = st.file_uploader("Upload file audio", type=['wav', 'mp3'])

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
            st.error(f"Gagal memproses file audio: {str(e)}")
            st.stop()

        st.write(f"Sample rate: {sr} Hz")
        st.write(f"Duration: {len(y)/sr:.2f} seconds")

        if sr != fs:
            st.warning(f"Resampling dari {sr} Hz ke {fs} Hz...")
            y = librosa.resample(y, orig_sr=sr, target_sr=fs)

        # Jika user memilih Tampilan Tiap Window, tampilkan semua window dengan hasil yang sama
        if window_type == "Tampilan Tiap Window":

            # Filter untuk satu window (default Kaiser) sebagai contoh
            filter_taps, _ = design_bandpass_filter(fs, lowcut, highcut, transition_width, stopband_attenuation, "Kaiser")
            y_filtered = signal.lfilter(filter_taps, 1.0, y)

            st.subheader("Grafik perbandingan Original Audio vs Filtered Audio")
            fig_time, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            t = np.linspace(0, len(y)/fs, len(y))
            axs[0].plot(t, y, color='blue')
            axs[0].set_title("Original Audio")
            axs[1].plot(t, y_filtered, color='green')
            axs[1].set_title("Filtered Audio (Kaiser Window)")
            axs[1].set_xlabel("Time (s)")
            st.pyplot(fig_time)

            st.subheader("Grafik Perbandingan Spektrum Frekuensi (FFT)")
            fig_fft, axs_fft = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            axs_fft[0].plot(np.fft.rfftfreq(len(y), 1/fs), np.abs(np.fft.rfft(y)) / len(y))
            axs_fft[0].set_title("FFT Original Audio")
            axs_fft[0].set_ylabel("Magnitude")
            axs_fft[1].plot(np.fft.rfftfreq(len(y_filtered), 1/fs), np.abs(np.fft.rfft(y_filtered)) / len(y_filtered))
            axs_fft[1].set_title("FFT Filtered Audio (Kaiser Window)")
            axs_fft[1].set_xlabel("Frequency (Hz)")
            axs_fft[1].set_ylabel("Magnitude")
            st.pyplot(fig_fft)

            st.subheader("Frequency Response dari Filter")
            st.markdown(f"Filter didesain untuk **{window_type.capitalize()}**")

            # Plot Amplitude & Phase Response per window
            fig_resp, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(12, 10), sharex=True, 
                gridspec_kw={'height_ratios': [2, 1]}  # ax1 akan 2x lebih tinggi dari ax2
            )

            for win in window_options:
                taps, _ = design_bandpass_filter(fs, lowcut, highcut, transition_width, stopband_attenuation, win)
                w, h = signal.freqz(taps, worN=8000, fs=fs)
                magnitude = 20 * np.log10(np.abs(h))
                phase = np.unwrap(np.angle(h)) * 180 / np.pi
                ax1.plot(w, magnitude, label=win)
                ax2.plot(w, phase, label=win)

            ax1.set_title("Amplitude Response tiap Window")
            ax1.set_ylabel("Magnitude (dB)")
            ax1.axvline(lowcut, color='gray', linestyle='--', linewidth=0.8)
            ax1.axvline(highcut, color='gray', linestyle='--', linewidth=0.8)
            ax1.grid()
            ax1.legend()

            ax2.set_title("Phase Response tiap Window")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Phase (degrees)")
            ax2.grid()
            ax2.legend()

            st.pyplot(fig_resp)
            st.stop()

        # Lanjut jika bukan Tampilan Tiap Window
        filter_taps, N = design_bandpass_filter(fs, lowcut, highcut, transition_width, stopband_attenuation, window_type)
        st.sidebar.markdown(f"*Filter Order*: {N}")

        y_filtered = signal.lfilter(filter_taps, 1.0, y)

        fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        t = np.linspace(0, len(y)/fs, len(y))
        axs[0].plot(t, y, color='blue')
        axs[0].set_title("Original Audio")
        axs[1].plot(t, y_filtered, color='green')
        axs[1].set_title("Filtered Audio")
        axs[1].set_xlabel("Time (s)")
        st.subheader("Grafik perbandingan Original Audio vs Filtered Audio")
        st.pyplot(fig)

        filtered_path = tmp_path.replace(".wav", "_filtered.wav").replace(".mp3", "_filtered.wav")
        sf.write(filtered_path, y_filtered, fs)

        st.subheader("Pemutaran Audio")
        st.write("Audio setelah filter:")
        st.audio(filtered_path, format='audio/wav')

        st.subheader("Grafik Perbandingan Spektrum Frekuensi (FFT)")
        fig_fft, axs_fft = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        axs_fft[0].plot(np.fft.rfftfreq(len(y), 1/fs), np.abs(np.fft.rfft(y)) / len(y))
        axs_fft[0].set_title("FFT Original Audio")
        axs_fft[0].set_ylabel("Magnitude")
        axs_fft[1].plot(np.fft.rfftfreq(len(y_filtered), 1/fs), np.abs(np.fft.rfft(y_filtered)) / len(y_filtered))
        axs_fft[1].set_title("FFT Filtered Audio")
        axs_fft[1].set_xlabel("Frequency (Hz)")
        axs_fft[1].set_ylabel("Magnitude")
        st.pyplot(fig_fft)

        st.subheader("Frequency Response dari Filter")
        st.markdown(f"Filter didesain menggunakan window: **{window_type.capitalize()}**")

        w, h = signal.freqz(filter_taps, worN=8000, fs=fs)
        amplitude_response = 20 * np.log10(np.abs(h))
        phase_response = np.unwrap(np.angle(h)) * 180 / np.pi

        fig_response, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(w, amplitude_response)
        ax1.set_title("Amplitude Response")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.axhline(-3, color='red', linestyle='--', label='-3 dB')
        ax1.axvline(lowcut, color='gray', linestyle='--')
        ax1.axvline(highcut, color='gray', linestyle='--')
        ax1.grid()
        ax1.legend()
        ax2.plot(w, phase_response)
        ax2.set_title("Phase Response")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.grid()
        st.pyplot(fig_response)

else:
    st.info("Silakan upload file audio (.wav / .mp3) untuk diproses.")
