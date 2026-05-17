import matplotlib.pyplot as plt
import numpy as np

def plot_sa_temperature(initial_temp=1.0, cooling_rate=0.99, min_temp=1e-3, steps=1000):
    temperatures = []
    current_temp = initial_temp
    
    for _ in range(steps):
        temperatures.append(current_temp)
        current_temp = max(current_temp * cooling_rate, min_temp)
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, color='#FF9800', linewidth=2, label=f'Cooling Rate: {cooling_rate}')
    plt.axhline(y=min_temp, color='r', linestyle='--', alpha=0.5, label=f'Min Temp: {min_temp}')
    
    plt.title('Simulated Annealing Temperature Schedule')
    plt.xlabel('Steps (Actions)')
    plt.ylabel('Temperature')
    plt.grid(alpha=0.3)
    plt.legend()
    
    output_file = 'sa_temperature_plot.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    # Using the default parameters from our SimulatedAnnealingAgent
    # (assuming initial_temp=1.0 and cooling_rate=0.99 for a typical run)
    plot_sa_temperature(initial_temp=1.0, cooling_rate=0.99, min_temp=1e-3, steps=1000)
