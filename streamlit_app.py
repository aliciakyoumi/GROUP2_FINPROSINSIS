import streamlit as st


st.title("Sinyal dan Sistem Biomedik")

st.header("Group 2 Final Project")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Text Input")
    # Text input allows users to enter text
    user_name = st.text_input("Enter your name:", "Student")
    st.write(f"Hello, {user_name}!")
    
    st.subheader("Number Input")
    # Number input allows users to enter numbers with optional constraints
    age = st.number_input("Enter your age:", min_value=0, max_value=120, value=20)
    st.write(f"You are {age} years old.")

with col2:
    st.subheader("Filter Frequency")
    # Sliders are great for selecting values from a range
    frequency = st.slider("Select a frequency (Hz):", 
                         min_value=20, 
                         max_value=20000, 
                         value=1000,
                         help="This is a slider to select frequency values")
    st.write(f"Selected frequency: {frequency} Hz")
    
    st.subheader("Selectbox")
    # Selectbox creates a dropdown menu
    option = st.selectbox("Choose a window function:",
                         ["Rectangular", "Hamming", "Hann", "Blackman", "Kaiser"])
    st.write(f"You selected: {option}")

# PART 3: INTERACTIVE ELEMENTS
# =============================================================================

st.header("Part 3: Interactive Elements")

# Checkbox creates a simple true/false input
show_plot = st.checkbox("Show a simple signal plot")

# This code only runs if the checkbox is checked
if show_plot:
    st.subheader("Signal Visualization")
    
    # Create a simple figure using matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Generate a simple sine wave
    fs = 1000  # Sample rate (Hz)
    duration = 1  # Duration (seconds)
    f = frequency  # Use the frequency from the slider
    
    # Generate time array
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Generate the signal
    signal = np.sin(2 * np.pi * f * t)
    
    ax.plot(t, signal)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Sine Wave ({f} Hz)')
    ax.grid(True)
    
    st.pyplot(fig)
    
    st.markdown("""
    **What's happening here?**
    
    1. We're generating a sine wave with the frequency you selected using the slider.
    2. We're using Matplotlib to create the plot.
    3. Streamlit's `st.pyplot()` function displays the Matplotlib figure in our app.
    """)
# PART 4: WORKING WITH DATA
# =============================================================================

st.header("Part 4: Working with Data")

# Create a simple dataframe
st.subheader("DataFrame Example")

# Create sample data for a filter
data = {
    'Coefficient Index': list(range(10)),
    'Value': [0.01, 0.05, 0.12, 0.20, 0.24, 0.20, 0.12, 0.05, 0.01, 0.0]
}

df = pd.DataFrame(data)

# Display the dataframe
st.dataframe(df)

# Create a simple bar chart
st.subheader("Filter Coefficients Visualization")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(df['Coefficient Index'], df['Value'])
ax.set_xlabel('Coefficient Index')
ax.set_ylabel('Value')
ax.set_title('FIR Filter Coefficients')
ax.grid(True, alpha=0.3)
st.pyplot(fig)
