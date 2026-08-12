#!/usr/bin/env python3
"""
Generate energy efficiency visualizations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up matplotlib style for publication
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass  # Use default style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})

def load_energy_data(json_path: Path) -> dict:
    """Load energy metrics from JSON file."""
    with open(json_path) as f:
        return json.load(f)

def plot_energy_comparison(data: dict, output_dir: Path):
    """Create bar chart comparing energy per query across methods."""
    measurements = data['measurements']
    
    methods = ['JIT Cascade', 'Random Sampling', 'L1 + OWL-ViT', 'Brute Force']
    method_keys = ['jit_cascade', 'random_sampling', 'l1_plus_owlvit', 'brute_force']
    
    energy_values = [measurements[k]['energy_per_query_joules'] for k in method_keys]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(methods, energy_values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, energy_values):
        height = bar.get_height()
        ax.annotate(f'{val:.0f} J',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Energy per Query (Joules)', fontweight='bold')
    ax.set_title('Energy Consumption Comparison\nObject Localization Query', fontweight='bold')
    ax.set_ylim(0, max(energy_values) * 1.15)
    
    # Add efficiency annotation
    jit_energy = energy_values[0]
    brute_energy = energy_values[3]
    ratio = brute_energy / jit_energy
    ax.annotate(f'JIT is {ratio:.0f}× more efficient',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'energy_comparison_bar.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved energy_comparison_bar.png/pdf")

def plot_power_profile(data: dict, output_dir: Path):
    """Create stacked bar chart showing power breakdown by operation type."""
    
    # Power consumption by operation type (Watts × duration)
    operations = ['CLIP\nEncoding', 'FAISS\nSearch', 'Depth\nProjection', 'DBSCAN\nClustering', 'OWL-ViT\nVerification']
    
    # Energy in Joules per query for each method (estimated breakdown)
    # Based on operation profiles
    jit_cascade = [0.6, 0.08, 3.0, 0.2, 72.0]      # 5 OWL-ViT calls
    l1_owlvit = [0.6, 0.08, 0, 0, 288.0]           # 20 OWL-ViT calls
    random = [0, 0, 0, 0, 288.0]                    # 20 OWL-ViT calls
    brute_force = [0, 0, 0, 0, 1440.0]             # 100 OWL-ViT calls
    
    x = np.arange(len(operations))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - 1.5*width, jit_cascade, width, label='JIT Cascade', color='#2ecc71')
    bars2 = ax.bar(x - 0.5*width, l1_owlvit, width, label='L1 + OWL-ViT', color='#9b59b6')
    bars3 = ax.bar(x + 0.5*width, random, width, label='Random Sampling', color='#3498db')
    bars4 = ax.bar(x + 1.5*width, brute_force, width, label='Brute Force', color='#e74c3c')
    
    ax.set_ylabel('Energy (Joules)', fontweight='bold')
    ax.set_title('Energy Breakdown by Operation Type', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(operations)
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 2000)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_breakdown_ops.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'energy_breakdown_ops.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved energy_breakdown_ops.png/pdf")

def plot_battery_life(data: dict, output_dir: Path):
    """Create grouped bar chart showing battery life for different robots."""
    robot_metrics = data['robot_metrics']
    
    robots = ['Fetch\n(800 Wh)', 'Spot\n(605 Wh)', 'TurtleBot\n(60 Wh)', 'Jetson AGX\n(40 Wh)']
    robot_keys = ['fetch', 'spot', 'turtlebot', 'jetson_agx']
    
    methods = ['jit_cascade', 'l1_plus_owlvit', 'random_sampling', 'brute_force']
    method_labels = ['JIT Cascade', 'L1 + OWL-ViT', 'Random Sampling', 'Brute Force']
    colors = ['#2ecc71', '#9b59b6', '#3498db', '#e74c3c']
    
    x = np.arange(len(robots))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        values = [robot_metrics[method][r]['battery_life_hours'] for r in robot_keys]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color)
    
    ax.set_ylabel('Battery Life (hours)', fontweight='bold')
    ax.set_title('Robot Battery Life During Continuous Object Localization', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(robots)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'battery_life_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'battery_life_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved battery_life_comparison.png/pdf")

def plot_queries_possible(data: dict, output_dir: Path):
    """Create chart showing total queries possible on a single charge."""
    robot_metrics = data['robot_metrics']
    
    # Focus on Fetch robot for clarity
    methods = ['JIT Cascade', 'L1 + OWL-ViT', 'Random Sampling', 'Brute Force']
    method_keys = ['jit_cascade', 'l1_plus_owlvit', 'random_sampling', 'brute_force']
    colors = ['#2ecc71', '#9b59b6', '#3498db', '#e74c3c']
    
    queries = [robot_metrics[k]['fetch']['total_queries_possible'] for k in method_keys]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(methods, queries, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, val in zip(bars, queries):
        height = bar.get_height()
        ax.annotate(f'{val:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Total Queries Possible', fontweight='bold')
    ax.set_title('Object Localization Queries per Battery Charge\nFetch Robot (800 Wh)', fontweight='bold')
    
    # Add improvement annotation
    ratio = queries[0] / queries[3]
    ax.annotate(f'JIT enables {ratio:.0f}× more queries',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'queries_per_charge.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'queries_per_charge.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved queries_per_charge.png/pdf")

def plot_cumulative_energy(output_dir: Path):
    """
    Create plot showing cumulative energy over time (simulated).
    This shows the key insight: JIT has low baseline with spikes, brute force constant high.
    """
    # Simulate 60 seconds of operation with periodic queries
    time = np.linspace(0, 60, 1000)
    
    # Idle power (Watts)
    idle_power = 25
    
    # JIT Cascade: Low idle, spike during query (~every 10s)
    jit_power = np.ones_like(time) * idle_power
    query_times = [5, 15, 25, 35, 45, 55]
    for qt in query_times:
        # Each query takes ~1 second at ~100W average
        mask = (time >= qt) & (time < qt + 1)
        jit_power[mask] = 100
    
    # Brute Force: Constant high power during queries (longer)
    brute_power = np.ones_like(time) * idle_power
    for qt in query_times:
        # Each query takes ~5 seconds at ~270W
        mask = (time >= qt) & (time < qt + 5.3)
        brute_power[mask] = 270
    
    # Cumulative energy (integrate power over time)
    dt = time[1] - time[0]
    jit_cumulative = np.cumsum(jit_power * dt)
    brute_cumulative = np.cumsum(brute_power * dt)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Power over time
    ax1.plot(time, jit_power, label='JIT Cascade', color='#2ecc71', linewidth=2)
    ax1.plot(time, brute_power, label='Brute Force', color='#e74c3c', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Power (Watts)', fontweight='bold')
    ax1.set_title('Power Consumption During Object Localization Queries', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 300)
    ax1.fill_between(time, jit_power, alpha=0.3, color='#2ecc71')
    ax1.fill_between(time, brute_power, alpha=0.2, color='#e74c3c')
    
    # Add query markers
    for qt in query_times:
        ax1.axvline(x=qt, color='gray', linestyle='--', alpha=0.5)
    
    # Cumulative energy
    ax2.plot(time, jit_cumulative/1000, label='JIT Cascade', color='#2ecc71', linewidth=2)
    ax2.plot(time, brute_cumulative/1000, label='Brute Force', color='#e74c3c', linewidth=2)
    ax2.set_xlabel('Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Cumulative Energy (kJ)', fontweight='bold')
    ax2.set_title('Cumulative Energy Consumption', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.fill_between(time, jit_cumulative/1000, alpha=0.3, color='#2ecc71')
    ax2.fill_between(time, brute_cumulative/1000, alpha=0.2, color='#e74c3c')
    
    # Add final ratio annotation
    final_ratio = brute_cumulative[-1] / jit_cumulative[-1]
    ax2.annotate(f'Brute Force uses {final_ratio:.1f}× more energy',
                xy=(0.65, 0.15), xycoords='axes fraction',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_energy_timeline.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cumulative_energy_timeline.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative_energy_timeline.png/pdf")

def plot_carbon_footprint(data: dict, output_dir: Path):
    """Create visualization of annual carbon footprint."""
    measurements = data['measurements']
    
    # Calculate annual CO2 (10,000 queries/day, 0.4 kg CO2/kWh)
    queries_per_year = 10000 * 365
    co2_per_kwh = 0.4  # kg
    
    methods = ['JIT Cascade', 'Random Sampling', 'L1 + OWL-ViT', 'Brute Force']
    method_keys = ['jit_cascade', 'random_sampling', 'l1_plus_owlvit', 'brute_force']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    annual_co2 = []
    for k in method_keys:
        energy_j = measurements[k]['energy_per_query_joules'] * queries_per_year
        energy_kwh = energy_j / 3600000
        co2_kg = energy_kwh * co2_per_kwh
        annual_co2.append(co2_kg)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(methods, annual_co2, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, val in zip(bars, annual_co2):
        height = bar.get_height()
        ax.annotate(f'{val:.0f} kg',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Annual CO₂ Emissions (kg)', fontweight='bold')
    ax.set_title('Carbon Footprint Comparison\n(10,000 queries/day)', fontweight='bold')
    
    # Add reduction annotation
    reduction = (1 - annual_co2[0] / annual_co2[3]) * 100
    ax.annotate(f'JIT reduces emissions by {reduction:.0f}%',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'carbon_footprint.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'carbon_footprint.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved carbon_footprint.png/pdf")

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / 'outputs/energy_analysis/energy_metrics.json'
    output_dir = base_dir / 'outputs/energy_analysis/figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {json_path}...")
    data = load_energy_data(json_path)
    
    print("Generating visualizations...")
    plot_energy_comparison(data, output_dir)
    plot_power_profile(data, output_dir)
    plot_battery_life(data, output_dir)
    plot_queries_possible(data, output_dir)
    plot_cumulative_energy(output_dir)
    plot_carbon_footprint(data, output_dir)
    
    print(f"\nAll figures saved to {output_dir}")

if __name__ == '__main__':
    main()
