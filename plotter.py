import os
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def parse_log_file(filename: str) -> Tuple[List[int], List[float], List[float]]:
    """Parse a log file and extract step numbers, validation losses and step times.
    
    Args:
        filename: Path to the log file
        
    Returns:
        Tuple containing lists of steps, validation losses and cumulative times
    """
    steps = []
    losses = []
    times = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Parse line using regex
            match = re.match(r'step:(\d+)/\d+ val_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms', line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                avg_time = float(match.group(4))
                
                steps.append(step)
                losses.append(loss)
                times.append(avg_time * step)  # Cumulative time
                
    return steps, losses, times

def create_plots(log_files: List[str], labels: List[str], output_dir: str = 'plots'):
    """Create and save validation visualization plots.
    
    Args:
        log_files: List of paths to log files
        labels: List of labels corresponding to each log file
        output_dir: Directory to save output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss vs steps
    plt.figure(figsize=(8, 6))
    for log_file, label in zip(log_files, labels):
        steps, losses, _ = parse_log_file(log_file)
        plt.plot(steps, losses, label=label, linewidth=2)
    plt.axhline(y=3.28, color='grey', linestyle=':', linewidth=2, label='Threshold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss vs Steps', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_vs_steps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot loss vs time
    plt.figure(figsize=(8, 6))
    for log_file, label in zip(log_files, labels):
        _, losses, times = parse_log_file(log_file)
        times = np.array(times) / (1000 * 60 * 60)  # ms to hours
        plt.plot(times, losses, label=label, linewidth=2)
    
    plt.axhline(y=3.28, color='r', linestyle=':', label='Threshold')
    plt.xlabel('Training Time (hours)', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss vs Time', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_vs_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Example usage
    log_files = [
        "records/110824_CastBf16/a833bed8-2fa8-4cfe-af05-58c1cc48bc30.txt",
        "records/110924_Unet/b096c044-a704-4779-9ada-290bdac74191.txt"
        # Add more log files here
    ]
    labels = [
        "Previous Record",
        "Attention UNet"
    ]
    create_plots(log_files, labels)
