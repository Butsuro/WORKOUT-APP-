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
fig, ax = plt.subplots() 
#Graph acceleration vs time
ax.plot(df["Time"], df["Ax"], label="Ax")
ax.plot(df["Time"], df["Ay"], label="Ay")
ax.plot(df["Time"], df["Az"], label="Az")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration")
ax.set_title("Sample Acceleration Data")
ax.legend()

plt.show()




import matplotlib.pyplot as plt
import numpy as np

# magnitude
magnitude = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)

# normalize bubble sizes (key fix)
sizes = 200 * (magnitude / magnitude.max())

plt.figure()

plt.scatter(df["Time"], magnitude,
            s=sizes,
            alpha=0.6,
            edgecolors='black')

plt.xlabel("Time (s)")
plt.ylabel("Acceleration Magnitude")
plt.title("Acceleration Magnitude vs Time")

plt.grid(True)

plt.show()