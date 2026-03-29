import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

data_folder = "rawData"

if not os.path.exists(data_folder):
    print("Folder not found:", data_folder)
    exit()

print("Starting storage process...")
print("Members found:", os.listdir(data_folder))

with h5py.File("project_data.h5", "w") as hdf5_file:

    raw_group = hdf5_file.create_group("raw")

    for member in os.listdir(data_folder):

        member_path = os.path.join(data_folder, member)

        if os.path.isdir(member_path):

            print("\nProcessing member:", member)

            member_group = raw_group.create_group(member)

            for file in os.listdir(member_path):

                if file.endswith(".csv"):

                    file_path = os.path.join(member_path, file)
                    print("Reading file:", file_path)

                    df = pd.read_csv(file_path)

                    df = df[
                        [
                            "Time (s)",
                            "Acceleration x (m/s^2)",
                            "Acceleration y (m/s^2)",
                            "Acceleration z (m/s^2)",
                        ]
                    ]
                    #renaming to shorter and clearer names
                    df.columns = ["Time", "Ax", "Ay", "Az"]

                    dataset_name = file.replace(".csv", "")
                    member_group.create_dataset(dataset_name, data=df.values)

print("\nHDF5 file created.")


#Step 3

def compute_magnitude(df):
    return np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)

def plot_acceleration_vs_time(ax, df, title, metadata):
    ax.plot(df["Time"], df["Ax"], label="Ax")
    ax.plot(df["Time"], df["Ay"], label="Ay")
    ax.plot(df["Time"], df["Az"], label="Az")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")
    ax.set_title(f"{title}\n{metadata}")
    ax.legend(fontsize=6)

def plot_bubble_chart(ax, df, title, metadata):
    df_5s = df[df["Time"] <= 5]
    ax.scatter(df_5s["Time"], df_5s["Magnitude"], s=df_5s["Magnitude"], alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"{title}\n{metadata}")

with h5py.File("project_data.h5", "r") as hdf5_file:
    raw = hdf5_file["raw"]

    for member in raw.keys():
        print("Plotting for:", member)
        files = list(raw[member].keys())

        fig, ax = plt.subplots(6, 2, figsize=(12, 18))
        fig.suptitle(f"Member: {member}", fontsize=16)

        for i, file in enumerate(files[:6]):
            dataset = raw[member][file]
            data = dataset[:]

            df = pd.DataFrame(data, columns=["Time", "Ax", "Ay", "Az"])
            df["Magnitude"] = compute_magnitude(df)

            #Meta data: file size
            shape = dataset.shape
            size_kb = dataset.nbytes / 1024
            metadata = f"Shape: {shape} | Size: {size_kb:.1f} KB"

            #acceleration vs time graph on left
            plot_acceleration_vs_time(
                ax[i, 0],
                df,
                title=f"{file} - Acceleration",
                metadata=metadata
            )

            #Bubble chart on the right
            plot_bubble_chart(
                ax[i, 1],
                df,
                title=f"{file} - Bubble",
                metadata=metadata
            )

        #plt.tight_layout()
        #plt.show()
        
# -----------------------------------------------------------------------
# Step 4 - Pre-processing
# -----------------------------------------------------------------------

from scipy.signal import butter, filtfilt

WINDOW_SIZE = 8       
HIGHPASS_CUTOFF = 0.3  
SAMPLE_RATE = 100       

def fill_missing(df):
    """Fill any missing values using forward fill then backward fill.
    Forward fill covers gaps mid-signal, backward fill handles NaNs at the start."""
    return df.ffill().bfill()

def apply_sma(df, window_size=WINDOW_SIZE):
    """Apply a Simple Moving Average filter to Ax, Ay, Az columns.
    Center=True means the window is centered around each point for minimal lag."""
    smoothed = df.copy()
    for col in ["Ax", "Ay", "Az"]:
        smoothed[col] = df[col].rolling(window=window_size, center=True).mean()
    # Rolling creates NaNs at edges — fill those too
    return smoothed.ffill().bfill()

def apply_highpass(df, cutoff=HIGHPASS_CUTOFF, fs=SAMPLE_RATE, order=4):
    """Apply a Butterworth high-pass filter to Ax, Ay, Az columns.
    Applied on top of the SMA output to remove any remaining gravity
    and low-frequency baseline drift that SMA alone cannot eliminate.
    - cutoff: frequency below which signals are removed (Hz)
    - fs: sample rate of the data (Hz)
    - order: filter sharpness — higher = steeper cutoff"""
    filtered = df.copy()
    # Normalized cutoff: must be between 0 and 1 (1 = Nyquist frequency)
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    # Design the Butterworth high-pass filter
    b, a = butter(order, normalized_cutoff, btype="high", analog=False)
    for col in ["Ax", "Ay", "Az"]:
        # filtfilt applies the filter twice (forward + backward) for zero phase distortion
        filtered[col] = filtfilt(b, a, df[col].values)
    return filtered

# Dictionary to hold plot samples per member: member -> list of (filename, raw_df, final_df)
# final_df = SMA then high-pass applied in sequence — this is what gets saved and plotted
member_plot_data = {}

# Open HDF5 in append mode to add the preprocessed group alongside raw
with h5py.File("project_data.h5", "a") as hdf5_file:

    # Create the top-level preprocessed group
    preprocessed_group = hdf5_file.create_group("preprocessed")

    # Loop through each member's folder
    for member in os.listdir(data_folder):
        member_path = os.path.join(data_folder, member)

        if os.path.isdir(member_path):
            print(f"Preprocessing: {member}")

            # Create a subgroup for this member under preprocessed
            member_group = preprocessed_group.create_group(member)
            member_plot_data[member] = []

            # Loop through each CSV file and apply preprocessing
            for file in os.listdir(member_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(member_path, file)

                    df_raw = pd.read_csv(file_path)
                    df_raw = df_raw[[
                        "Time (s)",
                        "Acceleration x (m/s^2)",
                        "Acceleration y (m/s^2)",
                        "Acceleration z (m/s^2)",
                    ]]
                    df_raw.columns = ["Time", "Ax", "Ay", "Az"]

                    # Step 4a: Fill any missing values
                    df_filled = fill_missing(df_raw)

                    # Step 4b: Apply SMA to reduce high-frequency noise
                    df_smoothed = apply_sma(df_filled)

                    # Step 4c: Apply high-pass filter directly on the SMA result
                    # The two filters are chained — high-pass receives SMA output as its input
                    df_final = apply_highpass(df_smoothed)

                    # Save the fully processed data (SMA + high-pass combined) into HDF5
                    dataset_name = file.replace(".csv", "")
                    member_group.create_dataset(dataset_name, data=df_final.values)

                    # Collect up to 6 files per member for plotting
                    if len(member_plot_data[member]) < 6:
                        member_plot_data[member].append((dataset_name, df_filled, df_final))

print("Preprocessed data saved to HDF5.")

# --- Step 4 Visualization: Raw vs Combined (SMA + High-Pass), first 5 seconds only ---
# One window per member, 6 rows x 2 cols (12 graphs: left=Raw, right=Combined)
colors = {"Ax": "red", "Ay": "green", "Az": "blue"}

for member, samples in member_plot_data.items():

    num_files = len(samples)  # up to 6 files
    fig, axes = plt.subplots(num_files, 2, figsize=(14, 3.5 * num_files))
    fig.suptitle(
        f"Step 4 - Pre-Processing | Member: {member}\n"
        f"Left = Raw  |  Right = SMA + High-Pass Combined  |  First 5s Shown",
        fontsize=13,
        fontweight="bold"
    )

    for row, (name, raw_df, final_df) in enumerate(samples):

        # Limit both signals to the first 5 seconds for display
        raw_5s   = raw_df[raw_df["Time"] <= 5]
        final_5s = final_df[final_df["Time"] <= 5]

        # Handle edge case where there is only 1 file (axes won't be 2D)
        ax_raw   = axes[row, 0] if num_files > 1 else axes[0]
        ax_final = axes[row, 1] if num_files > 1 else axes[1]

        # Left column: Raw data with all 3 axes overlaid
        for col, color in colors.items():
            ax_raw.plot(raw_5s["Time"], raw_5s[col], color=color, alpha=0.6, label=col)
        ax_raw.set_title(f"{name} — Raw (First 5s)")
        ax_raw.set_xlabel("Time (s)")
        ax_raw.set_ylabel("Acceleration")
        ax_raw.legend(fontsize=7)
        ax_raw.grid(True)

        # Right column: SMA + High-Pass combined result with all 3 axes overlaid
        for col, color in colors.items():
            ax_final.plot(final_5s["Time"], final_5s[col], color=color, linewidth=1.5, label=col)
        ax_final.set_title(f"{name} — SMA + High-Pass Combined (First 5s)")
        ax_final.set_xlabel("Time (s)")
        ax_final.set_ylabel("Acceleration")
        ax_final.legend(fontsize=7)
        ax_final.grid(True)

    plt.tight_layout()
    plt.show()
