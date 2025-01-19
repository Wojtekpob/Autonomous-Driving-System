import matplotlib.pyplot as plt
import csv
import argparse


def plot_mpc_data(file_path, tag=None):
    """
    Plots MPC control data from a CSV file.

    :param file_path: Path to the CSV file containing control data.
    :param tag: Optional tag to add to the plot title or output file.
    """
    time_steps = []
    delta_values = []
    a_values = []
    cte_values = []
    epsi_values = []

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            time_steps.append(int(row["time_step"]))
            delta_values.append(float(row["delta"]))
            a_values.append(float(row["a"]))
            cte_values.append(float(row["cte"]))
            epsi_values.append(float(row["epsi"]))

    plt.figure(figsize=(12, 8))

    plot_title = f"MPC Control Data"
    if tag:
        plot_title += f" - {tag}"

    plt.suptitle(plot_title, fontsize=16)

    plt.subplot(2, 2, 1)
    plt.plot(time_steps, delta_values, label="Delta (Steering Angle)")
    plt.title("Steering Angle (Delta)")
    plt.xlabel("Time Step")
    plt.ylabel("Delta (rad)")
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(time_steps, a_values, label="Throttle (a)", color="orange")
    plt.title("Throttle (a)")
    plt.xlabel("Time Step")
    plt.ylabel("Throttle")
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(time_steps, cte_values, label="CTE (Cross-Track Error)", color="green")
    plt.title("Cross-Track Error (CTE)")
    plt.xlabel("Time Step")
    plt.ylabel("CTE")
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(time_steps, epsi_values, label="EPSI (Orientation Error)", color="red")
    plt.title("Orientation Error (EPSI)")
    plt.xlabel("Time Step")
    plt.ylabel("EPSI")
    plt.grid()
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if tag:
        output_file = file_path.replace(".csv", f"_{tag}.png")
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MPC control data from a CSV file.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing control data.")
    parser.add_argument("--tag", type=str, help="Optional tag to add to the plot title and output file name.")

    args = parser.parse_args()

    plot_mpc_data(args.file_path, tag=args.tag)
