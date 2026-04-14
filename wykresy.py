import matplotlib.pyplot as plt


def parse_file(filename):
    data = {}

    with open(filename, 'r') as f:
        blocks = f.read().split('---')

        for block in blocks:
            if not block.strip():
                continue

            lines = block.strip().split('\n')
            entry = {}

            for line in lines:
                if ':' in line:
                    key, value = line.split(':')
                    entry[key.strip()] = value.strip()

            try:
                size = int(entry['Size'])
                time = float(entry['Time'])
                data[size] = time
            except KeyError:
                continue

    return data


def compute_speedup(data_t1, data_t8):
    sizes = sorted(set(data_t1.keys()) & set(data_t8.keys()))

    times_t1 = []
    times_t8 = []
    speedups = []

    for size in sizes:
        t1 = data_t1[size]
        t8 = data_t8[size]

        times_t1.append(t1)
        times_t8.append(t8)
        speedups.append(t1 / t8)

    return sizes, times_t1, times_t8, speedups


def plot_results(sizes, times_t8, speedups):

    plt.figure()
    plt.plot(sizes, times_t8, marker='o')
    plt.xlabel('Rozmiar danych')
    plt.ylabel('Czas (s)')
    plt.title('Czas wykonania (8 wątków)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('wykresy/czas.png')
    plt.close()


    plt.figure()
    plt.plot(sizes, speedups, marker='o')
    plt.xlabel('Rozmiar danych')
    plt.ylabel('Speedup (T1 / T8)')
    plt.title('Przyspieszenie')
    plt.grid()
    plt.tight_layout()
    plt.savefig('wykresy/speedup.png')
    plt.close()


if __name__ == "__main__":
    file_t1 = "wyniki/static_1000_t_1.txt"
    file_t8 = "wyniki/static_1000_t_8.txt"

    data_t1 = parse_file(file_t1)
    data_t8 = parse_file(file_t8)

    sizes, times_t1, times_t8, speedups = compute_speedup(data_t1, data_t8)

    plot_results(sizes, times_t8, speedups)