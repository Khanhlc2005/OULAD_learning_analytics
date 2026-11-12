
import time
import threading
import statistics
import psutil


def measure_function(func, repeats=3, n_items=None):
    """
    Đo thời gian, RAM, CPU và throughput của một hàm.

    func: Hàm cần đo.
    repeats: Số lần chạy thử.
    n_items: Số mẫu được xử lý, dùng để tính throughput.
    """

    if repeats < 3:
        raise ValueError("Benchmark cần chạy ít nhất 3 lần.")

    process = psutil.Process()
    results = []

    for run in range(repeats):

        # Ghi nhận RAM ban đầu
        peak_ram = [process.memory_info().rss / (1024 ** 2)]

        # Theo dõi RAM trong khi hàm chạy
        stop_monitor = threading.Event()

        def monitor_ram():
            while not stop_monitor.is_set():
                current_ram = process.memory_info().rss / (1024 ** 2)
                peak_ram[0] = max(peak_ram[0], current_ram)
                stop_monitor.wait(0.01)

        monitor = threading.Thread(target=monitor_ram)
        monitor.start()

        # Ghi nhận CPU và thời gian bắt đầu
        cpu_before = process.cpu_times()
        start_time = time.perf_counter()

        try:
            func()
        finally:
            elapsed = time.perf_counter() - start_time
            cpu_after = process.cpu_times()

            stop_monitor.set()
            monitor.join()

        # Tính CPU time
        cpu_time = (
            (cpu_after.user - cpu_before.user)
            + (cpu_after.system - cpu_before.system)
        )

        # CPU usage: phần trăm tương đương số lõi CPU sử dụng
        cpu_percent = (cpu_time / elapsed) * 100 if elapsed > 0 else 0

        # Tính throughput
        throughput = (
            n_items / elapsed
            if n_items is not None and elapsed > 0
            else None
        )

        results.append({
            "run": run + 1,
            "time_seconds": elapsed,
            "peak_ram_mb": peak_ram[0],
            "cpu_percent": cpu_percent,
            "throughput": throughput
        })

    return results


def summarize_benchmark(results):
    """
    Tổng hợp kết quả bằng median, min và max.
    """

    metrics = [
        "time_seconds",
        "peak_ram_mb",
        "cpu_percent",
        "throughput"
    ]

    summary = {}

    for metric in metrics:
        values = [
            row[metric]
            for row in results
            if row[metric] is not None
        ]

        if values:
            summary[metric] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values)
            }

    return summary
