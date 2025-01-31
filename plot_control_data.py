import matplotlib.pyplot as plt
import csv
import argparse


def plot_mpc_data(file_path, tag=None):
    """
    Plots MPC control data from a CSV file.

    :param file_path: Path to the CSV file containing control data.
    :param tag: Optional tag to add to the plot title or output file.
    """
    sample_indices = []  
    delta_values = []
    a_values = []
    cte_values = []
    epsi_values = []
    v_values = []
    desired_speed_values = []

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader):
            sample_indices.append(idx)  # Indeks próbki
            delta_values.append(float(row["delta"]))
            a_values.append(float(row["a"]))
            cte_values.append(float(row["cte"]))
            epsi_values.append(float(row["epsi"]))
            v_values.append(float(row["v"]))
            desired_speed_values.append(float(row["desired_speed"]))

    plt.figure(figsize=(14, 12))

    plt.subplot(3, 2, 1)
    plt.plot(sample_indices, delta_values, label="Delta (Kąt Skrętu)")
    plt.title("Kąt Skrętu (Delta)")
    plt.xlabel("Próbka")
    plt.ylabel("Delta (rad)")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(sample_indices, a_values, label="Przyspieszenie (a)", color="orange")
    plt.title("Przyspieszenie (a)")
    plt.xlabel("Próbka")
    plt.ylabel("Przyspieszenie")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(sample_indices, cte_values, label="CTE", color="green")
    plt.title("Błąd Boczny (CTE)")
    plt.xlabel("Próbka")
    plt.ylabel("CTE")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(sample_indices, epsi_values, label="EPSI", color="red")
    plt.title("Błąd Orientacji (EPSI)")
    plt.xlabel("Próbka")
    plt.ylabel("EPSI")
    plt.grid()
    plt.legend()

    plt.tight_layout(h_pad=2.5, rect=[0, 0, 1, 0.95])

    if tag:
        output_file = file_path.replace(".csv", f"_{tag}.png")
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Plot saved to {output_file}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MPC control data from a CSV file.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing control data.")
    parser.add_argument("--tag", type=str, help="Optional tag to add to the plot title and output file name.")

    args = parser.parse_args()

    plot_mpc_data(args.file_path, tag=args.tag)
