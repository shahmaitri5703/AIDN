import csv

def save_metrics_csv(metrics, filename):
    """
    metrics = list of dicts
    """
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
