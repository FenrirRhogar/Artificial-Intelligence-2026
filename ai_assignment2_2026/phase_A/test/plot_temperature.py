import matplotlib.pyplot as plt
import numpy as np

def plot_sa_temperature(initial_temp=0.01, cooling_rate=0.99, min_temp=1e-3, steps=1000):
    temperatures = []
    current_temp = initial_temp
    
    for _ in range(steps + 1):
        temperatures.append(current_temp)
        current_temp = max(current_temp * cooling_rate, min_temp)
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, color='#FF9800', linewidth=2.5, label=f'Cooling Rate: {cooling_rate}')
    plt.axhline(y=min_temp, color='r', linestyle='--', alpha=0.6, label=f'Min Temp: {min_temp}')
    
    plt.title('Simulated Annealing Temperature Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Temperature', fontsize=12)
    
    custom_ticks = [10, 100, 500, 1000]
    plt.xticks(custom_ticks)
    
    for tick in custom_ticks:
        plt.axvline(x=tick, color='gray', linestyle=':', alpha=0.4)
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_file = 'sa_temperature_plot.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    plot_sa_temperature(initial_temp=10.0, cooling_rate=0.99, min_temp=1e-3, steps=1000)
