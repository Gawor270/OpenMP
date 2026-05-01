import os
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


RESULTS_DIR = 'src/wyniki'
PLOTS_DIR = 'wykresy'

FILENAME_PATTERNS = [
    re.compile(r'^(?P<schedule>[a-zA-Z]+)_(?P<chunk>\d+)_t_(?P<threads>\d+)\.txt$'),
    re.compile(r'^(?P<schedule>[a-zA-Z]+)_(?P<chunk>\d+)_threads_(?P<threads>\d+)\.txt$'),
]


def parse_file_entries(filename):
    entries = []

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
                entries.append({
                    'size': size,
                    'time': time,
                })
            except KeyError:
                continue

    return entries


def aggregate_entries(entries):
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry['size'], []).append(entry['time'])

    result = {}
    for size, times in grouped.items():
        if not times:
            continue
        result[size] = {
            'mean': statistics.mean(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0.0,
            'n': len(times),
        }
    return result


def parse_filename(filename):
    for pattern in FILENAME_PATTERNS:
        match = pattern.match(filename)
        if match:
            return {
                'schedule': match.group('schedule').lower(),
                'chunk': int(match.group('chunk')),
                'threads': int(match.group('threads')),
            }
    return None


def discover_datasets(results_dir):
    datasets = {}

    for filename in sorted(os.listdir(results_dir)):
        metadata = parse_filename(filename)
        if metadata is None:
            continue

        path = os.path.join(results_dir, filename)
        entries = parse_file_entries(path)
        stats = aggregate_entries(entries)
        if not stats:
            continue

        schedule = metadata['schedule']
        chunk = metadata['chunk']
        threads = metadata['threads']

        datasets.setdefault(schedule, {}).setdefault(chunk, {})[threads] = {
            'filename': filename,
            'stats': stats,
        }

    return datasets


def chunk_score(chunk_data):
    score = 0
    for threads in (1, 8):
        thread_data = chunk_data.get(threads)
        if thread_data is None:
            continue
        score += sum(point['n'] for point in thread_data['stats'].values())
    return score


def collect_schedule_pairs(datasets):
    pairs = []

    for schedule, chunks in datasets.items():
        for chunk, chunk_data in chunks.items():
            if 1 not in chunk_data or 8 not in chunk_data:
                continue

            pairs.append({
                'schedule': schedule,
                'chunk': chunk,
                't1': chunk_data[1]['stats'],
                't8': chunk_data[8]['stats'],
                'file_t1': chunk_data[1]['filename'],
                'file_t8': chunk_data[8]['filename'],
            })

    pairs.sort(key=lambda item: (item['schedule'], item['chunk']))
    return pairs


def compute_speedup(stats_t1, stats_t8):
    sizes = sorted(set(stats_t1.keys()) & set(stats_t8.keys()))

    means_t1 = []
    stds_t1 = []
    means_t8 = []
    stds_t8 = []
    speedups = []

    for size in sizes:
        t1 = stats_t1[size]['mean']
        t8 = stats_t8[size]['mean']

        means_t1.append(t1)
        stds_t1.append(stats_t1[size]['std'])
        means_t8.append(t8)
        stds_t8.append(stats_t8[size]['std'])
        speedups.append(t1 / t8)

    return sizes, means_t1, stds_t1, means_t8, stds_t8, speedups


def plot_time_all_schedules(schedule_pairs):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure()
    for pair in schedule_pairs:
        sizes, means_t1, stds_t1, means_t8, stds_t8, _ = compute_speedup(pair['t1'], pair['t8'])
        if not sizes:
            continue

        label_base = f"{pair['schedule']} (chunk={pair['chunk']})"
        plt.errorbar(
            sizes,
            means_t1,
            yerr=stds_t1,
            marker='o',
            capsize=3,
            linestyle='--',
            label=f'{label_base}, t=1'
        )
        plt.errorbar(
            sizes,
            means_t8,
            yerr=stds_t8,
            marker='o',
            capsize=3,
            linestyle='-',
            label=f'{label_base}, t=8'
        )

    plt.xlabel('Rozmiar danych')
    plt.ylabel('Czas generowania (s)')
    plt.title('Czas generowania dla wszystkich schedule type')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'czas_all_schedules.png'))
    plt.close()


def plot_speedup_all_schedules(schedule_pairs):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure()
    for pair in schedule_pairs:
        sizes, _, _, _, _, speedups = compute_speedup(pair['t1'], pair['t8'])
        if not sizes:
            continue
        label_base = f"{pair['schedule']} (chunk={pair['chunk']})"
        plt.plot(
            sizes,
            speedups,
            marker='o',
            label=label_base
        )

    plt.xlabel('Rozmiar danych')
    plt.ylabel('Speedup (mean(T1) / mean(T8))')
    plt.title('Speedup dla wszystkich schedule type')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'speedup_all_schedules.png'))
    plt.close()


def print_summary(schedule_pairs):
    for pair in schedule_pairs:
        sizes, means_t1, _, means_t8, _, speedups = compute_speedup(pair['t1'], pair['t8'])
        print(f'=== {pair["schedule"]} (chunk={pair["chunk"]}) ===')
        print(f'files: t1={pair["file_t1"]}, t8={pair["file_t8"]}')
        for size, t1, t8, sp in zip(sizes, means_t1, means_t8, speedups):
            print(f'size={size}, t1_mean={t1:.9f}, t8_mean={t8:.9f}, speedup={sp:.3f}')
        print()


if __name__ == "__main__":
    datasets = discover_datasets(RESULTS_DIR)
    schedule_pairs = collect_schedule_pairs(datasets)

    if not schedule_pairs:
        raise RuntimeError('Brak par danych t=1 i t=8 dla żadnej konfiguracji.')

    plot_time_all_schedules(schedule_pairs)
    plot_speedup_all_schedules(schedule_pairs)
    print_summary(schedule_pairs)