print("Starting visualization generator from CSV files...")

import pandas as pd
print("Pandas imported successfully")

import matplotlib.pyplot as plt
print("Matplotlib imported successfully")

import numpy as np
print("Numpy imported successfully")

import os
print("OS imported successfully")

import glob
print("Glob imported successfully")

def load_simulation_results(results_dir="simulation_results"):
    """Load all simulation results from CSV files"""
    print(f"Loading simulation results from '{results_dir}' directory...")
    
    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found!")
        return None
    
    # Get all run files
    album_files = glob.glob(f"{results_dir}/run_*_album_results.csv")
    time_streams_files = glob.glob(f"{results_dir}/run_*_time_streams.csv")
    time_listeners_files = glob.glob(f"{results_dir}/run_*_time_listeners.csv")
    momentum_files = glob.glob(f"{results_dir}/run_*_momentum.csv")
    
    print(f"Found {len(album_files)} album result files")
    print(f"Found {len(time_streams_files)} time streams files")
    print(f"Found {len(time_listeners_files)} time listeners files")
    print(f"Found {len(momentum_files)} momentum files")
    
    if len(album_files) == 0:
        print("Error: No simulation result files found!")
        return None
    
    # Load and organize data
    results = []
    
    # Sort files to ensure consistent ordering
    album_files.sort()
    time_streams_files.sort()
    time_listeners_files.sort()
    momentum_files.sort()
    
    for i in range(len(album_files)):
        # Extract run number and seed from filename
        filename = os.path.basename(album_files[i])
        parts = filename.split('_')
        run_num = int(parts[1])
        seed = int(parts[3])
        
        print(f"Loading Run {run_num} (seed {seed})...")
        
        # Load album results
        album_df = pd.read_csv(album_files[i])
        
        # Load time streams
        time_streams_df = pd.read_csv(time_streams_files[i])
        
        # Load time listeners
        time_listeners_df = pd.read_csv(time_listeners_files[i])
        
        # Load momentum
        momentum_df = pd.read_csv(momentum_files[i])
        
        # Process and structure the data
        num_albums = len(album_df)
        
        # Extract gradual and immediate streams/listeners by album
        gradual_album_streams = album_df['Gradual_Album_Streams'].tolist()
        immediate_album_streams = album_df['Immediate_Album_Streams'].tolist()
        gradual_album_listeners = album_df['Gradual_Album_Listeners'].tolist()
        immediate_album_listeners = album_df['Immediate_Album_Listeners'].tolist()
        
        # Extract time series data
        gradual_time_streams = {}
        immediate_time_streams = {}
        gradual_time_listeners = {}
        immediate_time_listeners = {}
        gradual_momentum = {}
        immediate_momentum = {}
        
        for album_id in range(num_albums):
            # Time streams
            grad_streams_col = f'Gradual_Album_{album_id}_Streams'
            imm_streams_col = f'Immediate_Album_{album_id}_Streams'
            gradual_time_streams[album_id] = time_streams_df[grad_streams_col].tolist()
            immediate_time_streams[album_id] = time_streams_df[imm_streams_col].tolist()
            
            # Time listeners
            grad_listeners_col = f'Gradual_Album_{album_id}_Listeners'
            imm_listeners_col = f'Immediate_Album_{album_id}_Listeners'
            gradual_time_listeners[album_id] = time_listeners_df[grad_listeners_col].tolist()
            immediate_time_listeners[album_id] = time_listeners_df[imm_listeners_col].tolist()
            
            # Momentum
            grad_momentum_col = f'Gradual_Album_{album_id}_Momentum'
            imm_momentum_col = f'Immediate_Album_{album_id}_Momentum'
            gradual_momentum[album_id] = momentum_df[grad_momentum_col].tolist()
            immediate_momentum[album_id] = momentum_df[imm_momentum_col].tolist()
        
        # Create run data structure matching original format
        run_data = {
            'run': run_num,
            'seed': seed,
            'gradual_album_streams': gradual_album_streams,
            'immediate_album_streams': immediate_album_streams,
            'gradual_album_listeners': gradual_album_listeners,
            'immediate_album_listeners': immediate_album_listeners,
            'gradual_time_streams': gradual_time_streams,
            'immediate_time_streams': immediate_time_streams,
            'gradual_time_listeners': gradual_time_listeners,
            'immediate_time_listeners': immediate_time_listeners,
            'gradual_momentum': gradual_momentum,
            'immediate_momentum': immediate_momentum
        }
        
        results.append(run_data)
    
    print(f"Successfully loaded {len(results)} simulation runs!")
    return results

def create_comprehensive_visualizations(results):

    """Create all visualizations from loaded results"""
    print("Creating comprehensive visualizations...")
    
    num_albums = len(results[0]['gradual_album_streams'])
    num_runs = len(results)
    
    # Determine the number of periods and which is album 4 (zero fanbase)
    album_4_id = 1  # Album 4 has ID 1 (based on original code)
    first_album_id = list(results[0]['gradual_time_streams'].keys())[0]
    gradual_periods = len(results[0]['gradual_time_streams'][first_album_id])
    immediate_periods = len(results[0]['immediate_time_streams'][first_album_id])
    
    print(f"Detected {gradual_periods} gradual periods and {immediate_periods} immediate periods")
    print(f"Album 4 (zero fanbase) is at index {album_4_id}")
    
    # 1. Average streams per album by strategy
    print("\nCreating visualization 1: Average Streams per Album by Strategy")
    gradual_stream_means = []
    immediate_stream_means = []
    gradual_stream_stderrs = []
    immediate_stream_stderrs = []
    album_names = []

    for album_id in range(num_albums):
        # Calculate mean and standard error across all runs for each album
        gradual_streams = [run['gradual_time_streams'][album_id][-1] for run in results]
        immediate_streams = [run['immediate_time_streams'][album_id][-1] for run in results]
        
        gradual_stream_means.append(np.mean(gradual_streams))
        immediate_stream_means.append(np.mean(immediate_streams))
        gradual_stream_stderrs.append(np.std(gradual_streams) / np.sqrt(len(gradual_streams)))
        immediate_stream_stderrs.append(np.std(immediate_streams) / np.sqrt(len(immediate_streams)))
        
        if album_id == 0:
            album_names.append('Fanbase')
        else:
            album_names.append('No Fanbase')

    # Create DataFrame for streams
    df_album_streams = pd.DataFrame({
        'Album': album_names,
        'Gradual Release': gradual_stream_means,
        'Immediate Release': immediate_stream_means
    })

    # Plot streams with error bars
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(album_names))
    width = 0.35

    # Create bars with error bars
    ax.bar(x - width/2, gradual_stream_means, width, yerr=gradual_stream_stderrs, 
                color='skyblue', capsize=5, label='Gradual Release')
    ax.bar(x + width/2, immediate_stream_means, width, yerr=immediate_stream_stderrs, 
                color='lightcoral', capsize=5, label='Immediate Release')

    plt.ylabel('Average Streams', fontsize=20, fontweight='bold')
    plt.xlabel('Album', fontsize=20, fontweight='bold')
    ax.set_xticks(x, album_names, fontsize = 16, fontweight='bold')
    plt.legend(title='Strategy', fontsize=20, title_fontsize=22)

    # Better positioned value labels
    max_val = max(max(gradual_stream_means), max(immediate_stream_means))
    for i, (grad_val, imm_val, grad_err, imm_err) in enumerate(zip(gradual_stream_means, immediate_stream_means, gradual_stream_stderrs, immediate_stream_stderrs)):
        # Position labels above the error bars instead of just above the bars
        ax.text(i-width/2, grad_val + grad_err + max_val*0.01, f'{grad_val:.1f}', ha='center', fontsize=16, fontweight='bold')
        ax.text(i+width/2, imm_val + imm_err + max_val*0.01, f'{imm_val:.1f}', ha='center', fontsize=16, fontweight='bold')

    ax.margins(y=0.15)
    plt.tight_layout()
    plt.savefig('album_streams_comparison.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    print("DataFrame - Average Streams by Album:")
    print(df_album_streams.round(0))
    print()
    
    # 2. Average listeners per album by strategy
    print("Creating visualization 2: Average Listeners per Album by Strategy")
    
    gradual_listener_means = []
    immediate_listener_means = []
    gradual_listener_stderrs = []  # Added
    immediate_listener_stderrs = []  # Added
    
    for album_id in range(num_albums):
        # Calculate mean and standard error across all runs for each album
        gradual_listeners = [run['gradual_time_listeners'][album_id][-1] for run in results]
        immediate_listeners = [run['immediate_time_listeners'][album_id][-1] for run in results]
        
        gradual_listener_means.append(np.mean(gradual_listeners))
        immediate_listener_means.append(np.mean(immediate_listeners))
        gradual_listener_stderrs.append(np.std(gradual_listeners) / np.sqrt(len(gradual_listeners)))  # Added
        immediate_listener_stderrs.append(np.std(immediate_listeners) / np.sqrt(len(immediate_listeners)))  # Added
    
    # Create DataFrame for listeners
    df_album_listeners = pd.DataFrame({
        'Album': album_names,
        'Gradual Release': gradual_listener_means,
        'Immediate Release': immediate_listener_means
    })
    
    # Plot listeners with error bars
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(album_names))
    width = 0.35

    # Create bars with error bars
    ax.bar(x - width/2, gradual_listener_means, width, yerr=gradual_listener_stderrs, 
           color='lightgreen', capsize=5, label='Gradual Release')
    ax.bar(x + width/2, immediate_listener_means, width, yerr=immediate_listener_stderrs, 
           color='salmon', capsize=5, label='Immediate Release')

    ax.set_ylabel('Average Listeners', fontsize=20, fontweight='bold')
    ax.set_xlabel('Album', fontsize=20, fontweight='bold')
    ax.set_xticks(x, album_names, fontsize = 16, fontweight='bold')
    ax.legend(title='Strategy', fontsize=20, title_fontsize=22)

    max_val = max(max(gradual_listener_means), max(immediate_listener_means))
    for i, (grad_val, imm_val, grad_err, imm_err) in enumerate(zip(gradual_listener_means, immediate_listener_means, gradual_listener_stderrs, immediate_listener_stderrs)):
        ax.text(i-width/2, grad_val + grad_err + max_val*0.01, f'{grad_val:.1f}', ha='center', fontsize=16, fontweight='bold')
        ax.text(i+width/2, imm_val + imm_err + max_val*0.01, f'{imm_val:.1f}', ha='center', fontsize=16, fontweight='bold')

    ax.margins(y=0.15) 
    plt.tight_layout()
    plt.savefig('album_listener_comparison.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    print("DataFrame - Average Listeners by Album:")
    print(df_album_listeners.round(0))
    print()
    
    # 2.1 Listener per album by strategy per time period
    
    album_0_id = 1
    all_gradual_album0_runs = []
    all_immediate_album0_runs = []

    # Collect individual run data for Album 0
    for run in results:
        # Gradual data for Album 0
        gradual_album0_data = run['gradual_time_listeners'][album_0_id]
        all_gradual_album0_runs.append(gradual_album0_data)
        
        # Immediate data for Album 0
        immediate_album0_data = run['immediate_time_listeners'][album_0_id]
        all_immediate_album0_runs.append(immediate_album0_data)

    # Calculate means for each time period
    gradual_periods = len(all_gradual_album0_runs[0])
    immediate_periods = len(all_immediate_album0_runs[0])

    # Calculate means
    gradual_album0_means = [np.mean([run[period] for run in all_gradual_album0_runs]) 
                        for period in range(gradual_periods)]
    immediate_album0_means = [np.mean([run[period] for run in all_immediate_album0_runs]) 
                            for period in range(immediate_periods)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot individual runs with thin lines and low opacity
    gradual_time_periods = range(1, gradual_periods + 1)
    immediate_time_periods = range(1, immediate_periods + 1)

    for i, run_data in enumerate(all_gradual_album0_runs):
        ax.plot(gradual_time_periods, run_data, color='lightgreen', alpha=0.3, linewidth=1, 
                marker='o', markersize=3)

    for i, run_data in enumerate(all_immediate_album0_runs):
        ax.plot(immediate_time_periods, run_data, color='salmon', alpha=0.3, linewidth=1, 
                marker='s', markersize=3)

    # Plot mean lines with thick lines
    ax.plot(gradual_time_periods, gradual_album0_means, color='green', linewidth=4, 
            marker='o', markersize=8, label='Gradual Release (Mean)', 
            markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(immediate_time_periods, immediate_album0_means, color='indianred', linewidth=4, 
            marker='s', markersize=8, label='Immediate Release (Mean)', 
            markeredgecolor='maroon', markeredgewidth=2)

    # Add vertical line to show where immediate release ends
    ax.axvline(x=6, color='gray', linestyle=':', linewidth=2, alpha=0.7, 
            label=f'Immediate Release Ends (Period 6)')

    # Add value annotations for mean lines at key periods
    for i, period in enumerate(gradual_time_periods):
        if period <= immediate_periods or period == gradual_periods:
            # Gradual Release mean annotations
            ax.annotate(f'{gradual_album0_means[i]:.0f}', 
                        (period, gradual_album0_means[i]),
                        textcoords="offset points", xytext=(0,5), ha='center', 
                        fontsize=16, color='green', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    for i, period in enumerate(immediate_time_periods):
        # Immediate Release mean annotations
        ax.annotate(f'{immediate_album0_means[i]:.0f}', 
                    (period, immediate_album0_means[i]),
                    textcoords="offset points", xytext=(0,30), ha='center', 
                    fontsize=16, color='indianred', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    plt.ylabel('Total Listeners', fontsize=20, fontweight='bold')
    plt.xlabel('Time Period', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(1600, 2800)
    plt.legend(title='Strategy', loc='upper left', fontsize=18, title_fontsize=20)
    plt.tight_layout()
    plt.savefig('listeners_over_time_cum.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create DataFrame for Album 0 time series data
    max_periods = max(gradual_periods, immediate_periods)
    df_album0_time_streams = pd.DataFrame({
        'Time_Period': range(1, max_periods + 1),
        'Gradual_Album_0_Mean': gradual_album0_means + [np.nan] * (max_periods - gradual_periods),
        'Immediate_Album_0_Mean': immediate_album0_means + [np.nan] * (max_periods - immediate_periods)
    })

    print("DataFrame - Album 0 Streams Over Time:")
    print(df_album0_time_streams.round(0))
    print()

    # 2.2 Listeners (per time period values, not cumulative)

    # Process data for Album 0 only
    album_0_id = 1
    all_gradual_album0_runs = []
    all_immediate_album0_runs = []

    # Collect individual run data for Album 0
    for run in results:
        # Gradual data for Album 0 - convert from cumulative to per-period values
        gradual_cumulative = run['gradual_time_listeners'][album_0_id]
        gradual_album0_data = [gradual_cumulative[0]]  # First period stays the same
        for i in range(1, len(gradual_cumulative)):
            gradual_album0_data.append(gradual_cumulative[i] - gradual_cumulative[i-1])
        all_gradual_album0_runs.append(gradual_album0_data)
        
        # Immediate data for Album 0 - convert from cumulative to per-period values
        immediate_cumulative = run['immediate_time_listeners'][album_0_id]
        immediate_album0_data = [immediate_cumulative[0]]  # First period stays the same
        for i in range(1, len(immediate_cumulative)):
            immediate_album0_data.append(immediate_cumulative[i] - immediate_cumulative[i-1])
        all_immediate_album0_runs.append(immediate_album0_data)

    # Calculate means for each time period
    gradual_periods = len(all_gradual_album0_runs[0])
    immediate_periods = len(all_immediate_album0_runs[0])

    # Calculate means
    gradual_album0_means = [np.mean([run[period] for run in all_gradual_album0_runs]) 
                        for period in range(gradual_periods)]
    immediate_album0_means = [np.mean([run[period] for run in all_immediate_album0_runs]) 
                            for period in range(immediate_periods)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot individual runs with thin lines and low opacity
    gradual_time_periods = range(1, gradual_periods + 1)
    immediate_time_periods = range(1, immediate_periods + 1)

    for i, run_data in enumerate(all_gradual_album0_runs):
        ax.plot(gradual_time_periods, run_data, color='lightgreen', alpha=0.3, linewidth=1, 
                marker='o', markersize=3)

    for i, run_data in enumerate(all_immediate_album0_runs):
        ax.plot(immediate_time_periods, run_data, color='salmon', alpha=0.3, linewidth=1, 
                marker='s', markersize=3)

    # Plot mean lines with thick lines
    ax.plot(gradual_time_periods, gradual_album0_means, color='green', linewidth=4, 
            marker='o', markersize=8, label='Gradual Release (Mean)', 
            markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(immediate_time_periods, immediate_album0_means, color='indianred', linewidth=4, 
            marker='s', markersize=8, label='Immediate Release (Mean)', 
            markeredgecolor='maroon', markeredgewidth=2)

    # Add vertical line to show where immediate release ends
    ax.axvline(x=6, color='gray', linestyle=':', linewidth=2, alpha=0.7, 
            label=f'Immediate Release Ends (Period 6)')

    # Add value annotations for mean lines at key periods
    for i, period in enumerate(immediate_time_periods):
        # Immediate Release mean annotations
        ax.annotate(f'{immediate_album0_means[i]:.0f}', 
                    (period, immediate_album0_means[i]),
                    textcoords="offset points", xytext=(0,30), ha='center', 
                    fontsize=16, color='indianred', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    for i, period in enumerate(gradual_time_periods):
        if period <= immediate_periods or period == gradual_periods:
            # Gradual Release mean annotations
            ax.annotate(f'{gradual_album0_means[i]:.0f}', 
                        (period, gradual_album0_means[i]),
                        textcoords="offset points", xytext=(0,5), ha='center', 
                        fontsize=16, color='green', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    plt.ylabel('Total Listeners', fontsize=20, fontweight='bold')
    plt.xlabel('Time Period', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Strategy', loc='upper right', fontsize=18, title_fontsize=20)
    plt.tight_layout()
    ax.set_ylim(0, 3000)
    plt.savefig('listeners_over_time_per_period.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create DataFrame for Album 0 time series data
    max_periods = max(gradual_periods, immediate_periods)
    df_album0_time_streams = pd.DataFrame({
        'Time_Period': range(1, max_periods + 1),
        'Gradual_Album_0_Mean': gradual_album0_means + [np.nan] * (max_periods - gradual_periods),
        'Immediate_Album_0_Mean': immediate_album0_means + [np.nan] * (max_periods - immediate_periods)
    })

    print("DataFrame - Album 0 Streams Over Time:")
    print(df_album0_time_streams.round(0))
    print()

    # 3. Album 0 streams over time periods (individual runs + mean)
    print("Creating visualization 3: Album 0 Streams Over Time Periods (Individual Runs + Mean)")

    # Process data for Album 0 only
    album_0_id = 0
    all_gradual_album0_runs = []
    all_immediate_album0_runs = []

    # Collect individual run data for Album 0
    for run in results:
        # Gradual data for Album 0
        gradual_album0_data = run['gradual_time_streams'][album_0_id]
        all_gradual_album0_runs.append(gradual_album0_data)
        
        # Immediate data for Album 0
        immediate_album0_data = run['immediate_time_streams'][album_0_id]
        all_immediate_album0_runs.append(immediate_album0_data)

    # Calculate means for each time period
    gradual_periods = len(all_gradual_album0_runs[0])
    immediate_periods = len(all_immediate_album0_runs[0])

    # Calculate means
    gradual_album0_means = [np.mean([run[period] for run in all_gradual_album0_runs]) 
                        for period in range(gradual_periods)]
    immediate_album0_means = [np.mean([run[period] for run in all_immediate_album0_runs]) 
                            for period in range(immediate_periods)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot individual runs with thin lines and low opacity
    gradual_time_periods = range(1, gradual_periods + 1)
    immediate_time_periods = range(1, immediate_periods + 1)

    for i, run_data in enumerate(all_gradual_album0_runs):
        ax.plot(gradual_time_periods, run_data, color='skyblue', alpha=0.3, linewidth=1, 
                marker='o', markersize=3)

    for i, run_data in enumerate(all_immediate_album0_runs):
        ax.plot(immediate_time_periods, run_data, color='lightcoral', alpha=0.3, linewidth=1, 
                marker='s', markersize=3)

    # Plot mean lines with thick lines
    ax.plot(gradual_time_periods, gradual_album0_means, color='blue', linewidth=4, 
            marker='o', markersize=8, label='Gradual Release (Mean)', 
            markeredgecolor='darkblue', markeredgewidth=2)
    ax.plot(immediate_time_periods, immediate_album0_means, color='red', linewidth=4, 
            marker='s', markersize=8, label='Immediate Release (Mean)', 
            markeredgecolor='darkred', markeredgewidth=2)

    # Add vertical line to show where immediate release ends
    ax.axvline(x=6, color='gray', linestyle=':', linewidth=2, alpha=0.7, 
            label=f'Immediate Release Ends (Period 6)')

    # Add value annotations for mean lines at key periods
    for i, period in enumerate(gradual_time_periods):
        if period <= immediate_periods or period == gradual_periods:
            # Gradual Release mean annotations
            ax.annotate(f'{gradual_album0_means[i]:.0f}', 
                        (period, gradual_album0_means[i]),
                        textcoords="offset points", xytext=(0,20), ha='center', 
                        fontsize=16, color='darkblue', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    for i, period in enumerate(immediate_time_periods):
        # Immediate Release mean annotations
        ax.annotate(f'{immediate_album0_means[i]:.0f}', 
                    (period, immediate_album0_means[i]),
                    textcoords="offset points", xytext=(0,20), ha='center', 
                    fontsize=16, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    plt.ylabel('Total Streams', fontsize=20, fontweight='bold')
    plt.xlabel('Time Period', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Strategy', loc='upper left', fontsize=18, title_fontsize=20)
    plt.tight_layout()
    plt.savefig('streams_over_time_cum.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Create DataFrame for Album 0 time series data
    max_periods = max(gradual_periods, immediate_periods)
    df_album0_time_streams = pd.DataFrame({
        'Time_Period': range(1, max_periods + 1),
        'Gradual_Album_0_Mean': gradual_album0_means + [np.nan] * (max_periods - gradual_periods),
        'Immediate_Album_0_Mean': immediate_album0_means + [np.nan] * (max_periods - immediate_periods)
    })

    print("DataFrame - Album 0 Streams Over Time:")
    print(df_album0_time_streams.round(0))
    print()

    # 4. Album 0 streams over time periods (per time period values, not cumulative)

    # Process data for Album 0 only
    album_0_id = 0
    all_gradual_album0_runs = []
    all_immediate_album0_runs = []

    # Collect individual run data for Album 0
    for run in results:
        # Gradual data for Album 0 - convert from cumulative to per-period values
        gradual_cumulative = run['gradual_time_streams'][album_0_id]
        gradual_album0_data = [gradual_cumulative[0]]  # First period stays the same
        for i in range(1, len(gradual_cumulative)):
            gradual_album0_data.append(gradual_cumulative[i] - gradual_cumulative[i-1])
        all_gradual_album0_runs.append(gradual_album0_data)
        
        # Immediate data for Album 0 - convert from cumulative to per-period values
        immediate_cumulative = run['immediate_time_streams'][album_0_id]
        immediate_album0_data = [immediate_cumulative[0]]  # First period stays the same
        for i in range(1, len(immediate_cumulative)):
            immediate_album0_data.append(immediate_cumulative[i] - immediate_cumulative[i-1])
        all_immediate_album0_runs.append(immediate_album0_data)

    # Calculate means for each time period
    gradual_periods = len(all_gradual_album0_runs[0])
    immediate_periods = len(all_immediate_album0_runs[0])

    # Calculate means
    gradual_album0_means = [np.mean([run[period] for run in all_gradual_album0_runs]) 
                        for period in range(gradual_periods)]
    immediate_album0_means = [np.mean([run[period] for run in all_immediate_album0_runs]) 
                            for period in range(immediate_periods)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot individual runs with thin lines and low opacity
    gradual_time_periods = range(1, gradual_periods + 1)
    immediate_time_periods = range(1, immediate_periods + 1)

    for i, run_data in enumerate(all_gradual_album0_runs):
        ax.plot(gradual_time_periods, run_data, color='skyblue', alpha=0.3, linewidth=1, 
                marker='o', markersize=3)

    for i, run_data in enumerate(all_immediate_album0_runs):
        ax.plot(immediate_time_periods, run_data, color='lightcoral', alpha=0.3, linewidth=1, 
                marker='s', markersize=3)

    # Plot mean lines with thick lines
    ax.plot(gradual_time_periods, gradual_album0_means, color='blue', linewidth=4, 
            marker='o', markersize=8, label='Gradual Release (Mean)', 
            markeredgecolor='darkblue', markeredgewidth=2)
    ax.plot(immediate_time_periods, immediate_album0_means, color='red', linewidth=4, 
            marker='s', markersize=8, label='Immediate Release (Mean)', 
            markeredgecolor='darkred', markeredgewidth=2)

    # Add vertical line to show where immediate release ends
    ax.axvline(x=6, color='gray', linestyle=':', linewidth=2, alpha=0.7, 
            label=f'Immediate Release Ends (Period 6)')

    # Add value annotations for mean lines at key periods
    for i, period in enumerate(gradual_time_periods):
        if period <= immediate_periods or period == gradual_periods:
            # Gradual Release mean annotations
            ax.annotate(f'{gradual_album0_means[i]:.0f}', 
                        (period, gradual_album0_means[i]),
                        textcoords="offset points", xytext=(0,5), ha='center', 
                        fontsize=16, color='darkblue', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    for i, period in enumerate(immediate_time_periods):
        # Immediate Release mean annotations
        ax.annotate(f'{immediate_album0_means[i]:.0f}', 
                    (period, immediate_album0_means[i]),
                    textcoords="offset points", xytext=(0,20), ha='center', 
                    fontsize=16, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    plt.ylabel('Total Streams', fontsize=20, fontweight='bold')
    plt.xlabel('Time Period', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 6300)  # Set y-axis limit to match previous visualization
    plt.legend(title='Strategy', loc='upper right', fontsize=18, title_fontsize=20)
    plt.tight_layout()
    plt.savefig('streams_over_time_per_period.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Create DataFrame for Album 0 time series data
    max_periods = max(gradual_periods, immediate_periods)
    df_album0_time_streams = pd.DataFrame({
        'Time_Period': range(1, max_periods + 1),
        'Gradual_Album_0_Mean': gradual_album0_means + [np.nan] * (max_periods - gradual_periods),
        'Immediate_Album_0_Mean': immediate_album0_means + [np.nan] * (max_periods - immediate_periods)
    })

    print("DataFrame - Album 0 Streams Over Time:")
    print(df_album0_time_streams.round(0))
    print()

    # 5. Streams-to-Listeners Ratio per Album (UPDATED CALCULATION)
    print("Creating visualization 5: Average Streams per Listener per Album")
    gradual_ratio_means = []
    immediate_ratio_means = []
    gradual_ratio_stderrs = []
    immediate_ratio_stderrs = []

    for album_id in range(num_albums):
        # Store individual ratios for each run to calculate mean and std error
        gradual_ratios = []
        immediate_ratios = []
        
        for run in results:
            # Get the final period values from time series for this specific run and album
            grad_streams_final = run['gradual_time_streams'][album_id][-1]
            imm_streams_final = run['immediate_time_streams'][album_id][-1]
            grad_listeners_final = run['gradual_time_listeners'][album_id][-1]
            imm_listeners_final = run['immediate_time_listeners'][album_id][-1]
            
            # Only add ratios where listeners > 0
            if grad_listeners_final > 0:
                grad_ratio = grad_streams_final / grad_listeners_final
                gradual_ratios.append(grad_ratio)
                
            if imm_listeners_final > 0:
                imm_ratio = imm_streams_final / imm_listeners_final
                immediate_ratios.append(imm_ratio)
        
        # Calculate mean and standard error only from non-zero listener cases
        if gradual_ratios:  # Only calculate if we have valid ratios
            gradual_ratio_means.append(np.mean(gradual_ratios))
            gradual_ratio_stderrs.append(np.std(gradual_ratios) / np.sqrt(len(gradual_ratios)))
        else:
            gradual_ratio_means.append(0)  # or np.nan if you prefer
            gradual_ratio_stderrs.append(0)
            
        if immediate_ratios:  # Only calculate if we have valid ratios
            immediate_ratio_means.append(np.mean(immediate_ratios))
            immediate_ratio_stderrs.append(np.std(immediate_ratios) / np.sqrt(len(immediate_ratios)))
        else:
            immediate_ratio_means.append(0)  # or np.nan if you prefer
            immediate_ratio_stderrs.append(0)
    
    # Create DataFrame for ratios
    df_album_ratios = pd.DataFrame({
        'Album': album_names,
        'Gradual Release': gradual_ratio_means,
        'Immediate Release': immediate_ratio_means
    })
    
    # Plot ratios with error bars
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(album_names))
    width = 0.35

    # Create bars with error bars
    ax.bar(x - width/2, gradual_ratio_means, width, yerr=gradual_ratio_stderrs, 
           color='gold', capsize=5, label='Gradual Release')
    ax.bar(x + width/2, immediate_ratio_means, width, yerr=immediate_ratio_stderrs, 
           color='orange', capsize=5, label='Immediate Release')

    ax.set_ylabel('Streams per Listener', fontsize=20, fontweight='bold')
    ax.set_xlabel('Album', fontsize=20, fontweight='bold')
    ax.set_xticks(x, album_names, fontsize = 16, fontweight='bold')
    ax.legend(title='Strategy', fontsize=20, title_fontsize=22)

    max_val = max(max(gradual_ratio_means), max(immediate_ratio_means))
    for i, (grad_val, imm_val, grad_err, imm_err) in enumerate(zip(gradual_ratio_means, immediate_ratio_means, gradual_ratio_stderrs, immediate_ratio_stderrs)):
        ax.text(i-width/2, grad_val + grad_err + max_val*0.01, f'{grad_val:.1f}', ha='center', fontsize=16, fontweight='bold')
        ax.text(i+width/2, imm_val + imm_err + max_val*0.01, f'{imm_val:.1f}', ha='center', fontsize=16, fontweight='bold')
    
    ax.margins(y=0.15) 
    plt.tight_layout()
    plt.savefig('ratios.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    print("DataFrame - Streams per Listener by Album:")
    print(df_album_ratios.round(2))
    print()

    # 6. Momentum Metric Visualization (Album 0 only - following visualization #3 logic)
    print("Creating visualization 6: Album 0 Momentum Over Time Periods (Individual Runs + Mean)")

    # Process momentum data for Album 0 only
    album_0_id = 0
    all_gradual_album0_momentum_runs = []
    all_immediate_album0_momentum_runs = []

    # Collect individual run data for Album 0 momentum
    for run in results:
        # Gradual momentum data for Album 0 (cumulative sum over time)
        gradual_album0_momentum_data = np.cumsum(run['gradual_momentum'][album_0_id]).tolist()
        all_gradual_album0_momentum_runs.append(gradual_album0_momentum_data)
        
        # Immediate momentum data for Album 0 (cumulative sum over time)
        immediate_album0_momentum_data = np.cumsum(run['immediate_momentum'][album_0_id]).tolist()
        all_immediate_album0_momentum_runs.append(immediate_album0_momentum_data)

    # Calculate means for each time period
    gradual_periods = len(all_gradual_album0_momentum_runs[0])
    immediate_periods = len(all_immediate_album0_momentum_runs[0])

    # Calculate means
    gradual_album0_momentum_means = [np.mean([run[period] for run in all_gradual_album0_momentum_runs]) 
                                   for period in range(gradual_periods)]
    immediate_album0_momentum_means = [np.mean([run[period] for run in all_immediate_album0_momentum_runs]) 
                                     for period in range(immediate_periods)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot individual runs with thin lines and low opacity
    gradual_time_periods = range(1, gradual_periods + 1)
    immediate_time_periods = range(1, immediate_periods + 1)

    for i, run_data in enumerate(all_gradual_album0_momentum_runs):
        ax.plot(gradual_time_periods, run_data, color='orange', alpha=0.3, linewidth=1, 
                marker='o', markersize=3)

    for i, run_data in enumerate(all_immediate_album0_momentum_runs):
        ax.plot(immediate_time_periods, run_data, color='purple', alpha=0.3, linewidth=1, 
                marker='s', markersize=3)

    # Plot mean lines with thick lines
    ax.plot(gradual_time_periods, gradual_album0_momentum_means, color='orange', linewidth=4, 
            marker='o', markersize=8, label='Gradual Release (Mean)', 
            markeredgecolor='darkorange', markeredgewidth=2)
    ax.plot(immediate_time_periods, immediate_album0_momentum_means, color='purple', linewidth=4, 
            marker='s', markersize=8, label='Immediate Release (Mean)', 
            markeredgecolor='darkviolet', markeredgewidth=2)

    # Add vertical line to show where immediate release ends
    ax.axvline(x=6, color='gray', linestyle=':', linewidth=2, alpha=0.7, 
            label=f'Immediate Release Ends (Period 6)')

    # Add value annotations for mean lines at key periods (same font size as visualization #3)
    for i, period in enumerate(gradual_time_periods):
        if period <= immediate_periods or period == gradual_periods:
            # Gradual Release mean annotations
            ax.annotate(f'{gradual_album0_momentum_means[i]:.1f}',
                (period, gradual_album0_momentum_means[i]),
                textcoords="offset points", xytext=(0,10), ha='center',
                fontsize=16, color='darkorange', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    for i, period in enumerate(immediate_time_periods):
        # Immediate Release mean annotations
        ax.annotate(f'{immediate_album0_momentum_means[i]:.1f}', 
                    (period, immediate_album0_momentum_means[i]),
                    textcoords="offset points", xytext=(0,20), ha='center', 
                    fontsize=15, color='darkviolet', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    plt.ylabel('Cumulative Momentum Events', fontsize=20, fontweight='bold')
    plt.xlabel('Time Period', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Strategy', loc='upper left', fontsize=18, title_fontsize=20)
    plt.tight_layout()
    plt.savefig('momentum_over_time.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Create DataFrame for Album 0 momentum time series data
    max_periods = max(gradual_periods, immediate_periods)
    df_album0_momentum_time = pd.DataFrame({
        'Time_Period': range(1, max_periods + 1),
        'Gradual_Album_0_Momentum_Mean': gradual_album0_momentum_means + [np.nan] * (max_periods - gradual_periods),
        'Immediate_Album_0_Momentum_Mean': immediate_album0_momentum_means + [np.nan] * (max_periods - immediate_periods)
    })

    print("DataFrame - Album 0 Momentum Over Time:")
    print(df_album0_momentum_time.round(1))
    print()
    
    # 6.5 Momentum Metric Visualization (Per period values, not cumulative)

    # Process momentum data for Album 0 only
    album_0_id = 0
    all_gradual_album0_momentum_runs = []
    all_immediate_album0_momentum_runs = []

    # Collect individual run data for Album 0 momentum
    for run in results:
        # Gradual momentum data for Album 0 (per period, not cumulative)
        gradual_album0_momentum_data = run['gradual_momentum'][album_0_id]
        all_gradual_album0_momentum_runs.append(gradual_album0_momentum_data)
        
        # Immediate momentum data for Album 0 (per period, not cumulative)
        immediate_album0_momentum_data = run['immediate_momentum'][album_0_id]
        all_immediate_album0_momentum_runs.append(immediate_album0_momentum_data)

    # Calculate means for each time period
    gradual_periods = len(all_gradual_album0_momentum_runs[0])
    immediate_periods = len(all_immediate_album0_momentum_runs[0])

    # Calculate means
    gradual_album0_momentum_means = [np.mean([run[period] for run in all_gradual_album0_momentum_runs]) 
                                   for period in range(gradual_periods)]
    immediate_album0_momentum_means = [np.mean([run[period] for run in all_immediate_album0_momentum_runs]) 
                                     for period in range(immediate_periods)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot individual runs with thin lines and low opacity
    gradual_time_periods = range(1, gradual_periods + 1)
    immediate_time_periods = range(1, immediate_periods + 1)

    for i, run_data in enumerate(all_gradual_album0_momentum_runs):
        ax.plot(gradual_time_periods, run_data, color='orange', alpha=0.3, linewidth=1, 
                marker='o', markersize=3)

    for i, run_data in enumerate(all_immediate_album0_momentum_runs):
        ax.plot(immediate_time_periods, run_data, color='purple', alpha=0.3, linewidth=1, 
                marker='s', markersize=3)

    # Plot mean lines with thick lines
    ax.plot(gradual_time_periods, gradual_album0_momentum_means, color='orange', linewidth=4, 
            marker='o', markersize=8, label='Gradual Release (Mean)', 
            markeredgecolor='darkorange', markeredgewidth=2)
    ax.plot(immediate_time_periods, immediate_album0_momentum_means, color='purple', linewidth=4, 
            marker='s', markersize=8, label='Immediate Release (Mean)', 
            markeredgecolor='darkviolet', markeredgewidth=2)

    # Add vertical line to show where immediate release ends
    ax.axvline(x=6, color='gray', linestyle=':', linewidth=2, alpha=0.7, 
            label=f'Immediate Release Ends (Period 6)')

    for i, period in enumerate(immediate_time_periods):
        # Immediate Release mean annotations
        ax.annotate(f'{immediate_album0_momentum_means[i]:.1f}', 
                    (period, immediate_album0_momentum_means[i]),
                    textcoords="offset points", xytext=(0,32), ha='center', 
                    fontsize=15, color='darkviolet', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
 
    # Add value annotations for mean lines at key periods (same font size as visualization #3)
    for i, period in enumerate(gradual_time_periods):
        if period <= immediate_periods or period == gradual_periods:
            # Gradual Release mean annotations
            ax.annotate(f'{gradual_album0_momentum_means[i]:.1f}',
                (period, gradual_album0_momentum_means[i]),
                textcoords="offset points", xytext=(0,5), ha='center',
                fontsize=16, color='darkorange', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

    plt.ylabel('Momentum Events (Per Period)', fontsize=20, fontweight='bold')
    plt.xlabel('Time Period', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Strategy', loc='upper right', fontsize=18, title_fontsize=20)
    plt.tight_layout()
    ax.set_ylim(0, 18500000)
    plt.savefig('momentum_per_period.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Create DataFrame for Album 0 momentum time series data
    max_periods = max(gradual_periods, immediate_periods)
    df_album0_momentum_time = pd.DataFrame({
        'Time_Period': range(1, max_periods + 1),
        'Gradual_Album_0_Momentum_Mean': gradual_album0_momentum_means + [np.nan] * (max_periods - gradual_periods),
        'Immediate_Album_0_Momentum_Mean': immediate_album0_momentum_means + [np.nan] * (max_periods - immediate_periods)
    })

    print("DataFrame - Album 0 Momentum Per Period:")
    print(df_album0_momentum_time.round(1))
    print()

    # 7. Momentum by Album (with error bars)

    # Calculate average momentum by album and strategy
    grad_momentum_events_means = []
    imm_momentum_events_means = []
    gradual_momentum_stderrs = []  # Added
    immediate_momentum_stderrs = []  # Added
    
    for album_id in range(num_albums):

        grad_momentum_events = [sum(run['gradual_momentum'][album_id]) for run in results]
        imm_momentum_events = [sum(run['immediate_momentum'][album_id]) for run in results]
        
        grad_momentum_events_means.append(np.mean(grad_momentum_events))
        imm_momentum_events_means.append(np.mean(imm_momentum_events))
        gradual_momentum_stderrs.append(np.std(grad_momentum_events) / np.sqrt(len(grad_momentum_events)))  # Added
        immediate_momentum_stderrs.append(np.std(imm_momentum_events) / np.sqrt(len(imm_momentum_events)))  # Added
    
    # Create DataFrame for momentum by album
    df_momentum_album = pd.DataFrame({
        'Album': album_names,
        'Gradual Release': grad_momentum_events_means,
        'Immediate Release': imm_momentum_events_means
    })
    
    # Plot momentum by album with error bars
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(album_names))
    width = 0.35

    # Create bars with error bars
    ax.bar(x - width/2, grad_momentum_events_means, width, yerr=gradual_momentum_stderrs, 
           color='orange', capsize=5, label='Gradual Release')
    ax.bar(x + width/2, imm_momentum_events_means, width, yerr=immediate_momentum_stderrs, 
           color='purple', capsize=5, label='Immediate Release')

    ax.set_ylabel('Total Momentum Events', fontsize=20, fontweight='bold')
    ax.set_xlabel('Album', fontsize=20, fontweight='bold')
    ax.set_xticks(x, album_names, fontsize = 16, fontweight='bold')
    ax.legend(title='Strategy', fontsize=20, title_fontsize=22)

    max_val = max(max(grad_momentum_events_means), max(imm_momentum_events_means))
    for i, (grad_val, imm_val, grad_err, imm_err) in enumerate(zip(grad_momentum_events_means, imm_momentum_events_means, gradual_momentum_stderrs, immediate_momentum_stderrs)):
        ax.text(i-width/2, grad_val + grad_err + max_val*0.01, f'{grad_val:.1f}', ha='center', fontsize=16, fontweight='bold')
        ax.text(i+width/2, imm_val + imm_err + max_val*0.01, f'{imm_val:.1f}', ha='center', fontsize=16, fontweight='bold')
    
    ax.margins(y=0.15) 
    plt.tight_layout()
    plt.savefig('momentum_means.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    print("DataFrame - Momentum by Album:")
    print(df_momentum_album.round(1))
    print()
    
    return (df_album_streams, df_album_listeners,
            df_album_ratios, df_momentum_album)

def create_average_streams_visualization(results):
    """Create visualization showing average gradual vs immediate streams across all runs"""
    print("Creating Average Streams Across Runs Visualization...")
    
    num_albums = len(results[0]['gradual_album_streams'])
    num_runs = len(results)
    
    # Collect streams data for each album across all runs
    gradual_streams_by_album = [[] for _ in range(num_albums)]
    immediate_streams_by_album = [[] for _ in range(num_albums)]
    
    for run in results:
        for album_id in range(num_albums):
            gradual_streams_by_album[album_id].append(run['gradual_album_streams'][album_id])
            immediate_streams_by_album[album_id].append(run['immediate_album_streams'][album_id])
    
    # Calculate means and standard deviations
    gradual_means = [np.mean(streams) for streams in gradual_streams_by_album]
    immediate_means = [np.mean(streams) for streams in immediate_streams_by_album]
    gradual_stds = [np.std(streams) for streams in gradual_streams_by_album]
    immediate_stds = [np.std(streams) for streams in immediate_streams_by_album]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Bar chart with error bars
    album_names = [f'Album {i}' for i in range(num_albums)]
    x_pos = np.arange(len(album_names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, gradual_means, width, yerr=gradual_stds, 
                    label='Gradual Release', color='skyblue', alpha=0.8, capsize=5)
    bars2 = ax1.bar(x_pos + width/2, immediate_means, width, yerr=immediate_stds,
                    label='Immediate Release', color='lightcoral', alpha=0.8, capsize=5)
    
    ax1.set_xlabel('Album')
    ax1.set_ylabel('Average Streams')
    ax1.set_title(f'Average Streams per Album Across {num_runs} Runs\n(with Standard Deviation)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(album_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (grad_mean, imm_mean, grad_std, imm_std) in enumerate(zip(gradual_means, immediate_means, gradual_stds, immediate_stds)):
        ax1.text(i - width/2, grad_mean + grad_std + 0.5, f'{grad_mean:.0f}±{grad_std:.0f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i + width/2, imm_mean + imm_std + 0.5, f'{imm_mean:.0f}±{imm_std:.0f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Right plot: Evolution of running mean and individual album performance
    run_numbers = list(range(1, num_runs + 1))
    
    # Calculate running means for total streams
    gradual_totals_per_run = []
    immediate_totals_per_run = []
    gradual_running_means = []
    immediate_running_means = []
    
    for i, run in enumerate(results):
        gradual_total = sum(run['gradual_album_streams'])
        immediate_total = sum(run['immediate_album_streams'])
        gradual_totals_per_run.append(gradual_total)
        immediate_totals_per_run.append(immediate_total)
        
        # Calculate running mean up to current run
        gradual_running_mean = np.mean(gradual_totals_per_run[:i+1])
        immediate_running_mean = np.mean(immediate_totals_per_run[:i+1])
        gradual_running_means.append(gradual_running_mean)
        immediate_running_means.append(immediate_running_mean)
    
    # Plot individual run totals (thin lines)
    ax2.plot(run_numbers, gradual_totals_per_run, color='skyblue', linewidth=2, alpha=0.7,
             marker='o', markersize=6, label='Gradual - Individual Runs', markeredgecolor='darkblue', markeredgewidth=1)
    ax2.plot(run_numbers, immediate_totals_per_run, color='lightcoral', linewidth=2, alpha=0.7,
             marker='s', markersize=6, label='Immediate - Individual Runs', markeredgecolor='darkred', markeredgewidth=1)
    
    # Plot running means (thick lines)
    ax2.plot(run_numbers, gradual_running_means, color='darkblue', linewidth=4, 
             marker='D', markersize=8, label='Gradual - Running Mean', markeredgecolor='white', markeredgewidth=2)
    ax2.plot(run_numbers, immediate_running_means, color='darkred', linewidth=4, 
             marker='D', markersize=8, label='Immediate - Running Mean', markeredgecolor='white', markeredgewidth=2)
    
    # Add final mean value annotations
    final_grad_mean = gradual_running_means[-1]
    final_imm_mean = immediate_running_means[-1]
    
    ax2.annotate(f'Final Mean: {final_grad_mean:.0f}', 
                xy=(num_runs, final_grad_mean), xytext=(num_runs-0.3, final_grad_mean+50),
                ha='center', fontsize=9, color='darkblue', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    ax2.annotate(f'Final Mean: {final_imm_mean:.0f}', 
                xy=(num_runs, final_imm_mean), xytext=(num_runs-0.3, final_imm_mean-50),
                ha='center', fontsize=9, color='darkred', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Total Streams')
    ax2.set_title(f'Total Streams Evolution\n(Individual Runs & Running Means)', fontweight='bold')
    ax2.set_xticks(run_numbers)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('means_over_time.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Album': album_names,
        'Gradual_Mean': gradual_means,
        'Gradual_Std': gradual_stds,
        'Immediate_Mean': immediate_means,
        'Immediate_Std': immediate_stds,
        'Difference_Mean': [imm - grad for grad, imm in zip(gradual_means, immediate_means)],
        'Gradual_Better': [grad > imm for grad, imm in zip(gradual_means, immediate_means)]
    })
    
    print("Summary Statistics Across All Runs:")
    print(summary_df.round(1))
    print()
    
    # Overall totals
    total_gradual_mean = sum(gradual_means)
    total_immediate_mean = sum(immediate_means)
    print(f"Total Streams - Gradual: {total_gradual_mean:.1f}, Immediate: {total_immediate_mean:.1f}")
    print(f"Overall Difference: {total_immediate_mean - total_gradual_mean:.1f} streams")
    print(f"{'Immediate' if total_immediate_mean > total_gradual_mean else 'Gradual'} strategy performs better overall")
    
    return summary_df

def create_album_performance_by_runs_visualization(results):
    """Create detailed visualization showing individual album performance across runs"""
    print("Creating Album-by-Album Performance Across Runs Visualization...")
    
    num_albums = len(results[0]['gradual_album_streams'])
    num_runs = len(results)
    run_numbers = list(range(1, num_runs + 1))
    
    # Create subplot grid: 2 rows (gradual/immediate), num_albums columns
    fig, axes = plt.subplots(2, num_albums, figsize=(30, 19))
    if num_albums == 1:
        axes = axes.reshape(2, 1)
    
    for album_id in range(num_albums):
        # Collect data for this album across all runs
        gradual_album_streams = [run['gradual_album_streams'][album_id] for run in results]
        immediate_album_streams = [run['immediate_album_streams'][album_id] for run in results]
        
        # Calculate running means
        gradual_running_means = []
        immediate_running_means = []
        
        for i in range(num_runs):
            gradual_running_means.append(np.mean(gradual_album_streams[:i+1]))
            immediate_running_means.append(np.mean(immediate_album_streams[:i+1]))
        
        # Gradual Release subplot (top row)
        ax_grad = axes[0, album_id]
        
        # Plot individual runs (light blue, thin line)
        ax_grad.plot(run_numbers, gradual_album_streams, color='lightblue', 
                    linewidth=2, alpha=0.7, marker='o', markersize=6, 
                    label='Individual Runs', markeredgecolor='white', markeredgewidth=1)
        
        # Plot running mean (dark blue, thick line)
        ax_grad.plot(run_numbers, gradual_running_means, color='darkblue', linewidth=4,
                    marker='D', markersize=8, label='Running Mean', 
                    markeredgecolor='white', markeredgewidth=2)
        
        # Add final mean annotation
        final_mean = gradual_running_means[-1]
        ax_grad.annotate(f'Mean: {final_mean:.0f}', 
                        xy=(num_runs, final_mean), xytext=(num_runs-0.3, final_mean),
                        ha='center', fontsize=10, color='darkblue', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        ax_grad.set_title(f'Album {album_id} - Gradual Release', fontweight='bold', fontsize=12)
        ax_grad.set_xlabel('Run Number')
        ax_grad.set_ylabel('Streams')
        ax_grad.grid(True, alpha=0.3)
        ax_grad.set_xticks(run_numbers)
        if album_id == 0:  # Only show legend on first subplot
            ax_grad.legend(loc='upper left', fontsize=9)
        
        # Immediate Release subplot (bottom row)
        ax_imm = axes[1, album_id]
        
        # Plot individual runs (light red, thin line)
        ax_imm.plot(run_numbers, immediate_album_streams, color='lightcoral', 
                   linewidth=2, alpha=0.7, marker='s', markersize=6, 
                   label='Individual Runs', markeredgecolor='white', markeredgewidth=1)
        
        # Plot running mean (dark red, thick line)
        ax_imm.plot(run_numbers, immediate_running_means, color='darkred', linewidth=4,
                   marker='D', markersize=8, label='Running Mean', 
                   markeredgecolor='white', markeredgewidth=2)
        
        # Add final mean annotation
        final_mean = immediate_running_means[-1]
        ax_imm.annotate(f'Mean: {final_mean:.0f}', 
                       xy=(num_runs, final_mean), xytext=(num_runs-0.3, final_mean),
                       ha='center', fontsize=10, color='darkred', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
        
        ax_imm.set_title(f'Album {album_id} - Immediate Release', fontweight='bold', fontsize=12)
        ax_imm.set_xlabel('Run Number')
        ax_imm.set_ylabel('Streams')
        ax_imm.grid(True, alpha=0.3)
        ax_imm.set_xticks(run_numbers)
        if album_id == 0:  # Only show legend on first subplot
            ax_imm.legend(loc='upper left', fontsize=9)
    
    plt.suptitle(f'Individual Album Performance Across {num_runs} Runs\n(Individual Results and Running Means)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('means_over_time2.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    # Create summary dataframe for album-by-album analysis
    album_summary_data = []
    for album_id in range(num_albums):
        gradual_streams = [run['gradual_album_streams'][album_id] for run in results]
        immediate_streams = [run['immediate_album_streams'][album_id] for run in results]
        
        album_summary_data.append({
            'Album': f'Album {album_id}',
            'Gradual_Mean': np.mean(gradual_streams),
            'Gradual_Std': np.std(gradual_streams),
            'Gradual_Min': np.min(gradual_streams),
            'Gradual_Max': np.max(gradual_streams),
            'Immediate_Mean': np.mean(immediate_streams),
            'Immediate_Std': np.std(immediate_streams),
            'Immediate_Min': np.min(immediate_streams),
            'Immediate_Max': np.max(immediate_streams),
            'Difference': np.mean(immediate_streams) - np.mean(gradual_streams),
            'Better_Strategy': 'Immediate' if np.mean(immediate_streams) > np.mean(gradual_streams) else 'Gradual'
        })
    
    album_summary_df = pd.DataFrame(album_summary_data)
    
    print("Detailed Album Performance Summary:")
    print(album_summary_df.round(1))
    print()
    
    return album_summary_df

def generate_all_visualizations(results_dir="simulation_results"):
    """Main function to load data and generate all visualizations"""
    print("=" * 80)
    print("MUSIC ALBUM RELEASE STRATEGY ANALYSIS")
    print("Visualization Generator from CSV Results")
    print("=" * 80)
    
    # Load simulation results
    results = load_simulation_results(results_dir)
    
    if results is None:
        print("Failed to load simulation results. Please check that:")
        print(f"1. The '{results_dir}' directory exists")
        print("2. CSV files from simulation runs are present")
        print("3. File naming follows the pattern: run_X_seed_Y_[type].csv")
        return None
    
    num_runs = len(results)
    num_albums = len(results[0]['gradual_album_streams'])
    
    print(f"\nSuccessfully loaded {num_runs} simulation runs with {num_albums} albums each")
    print()
    
    # Create the new average streams visualization first
    print("=" * 60)
    print("AVERAGE STREAMS ACROSS RUNS ANALYSIS")
    print("=" * 60)
    average_streams_summary = create_average_streams_visualization(results)
    
    # Create detailed album-by-album analysis
    print("\n" + "=" * 60)
    print("ALBUM-BY-ALBUM PERFORMANCE ANALYSIS")
    print("=" * 60)
    album_performance_summary = create_album_performance_by_runs_visualization(results)
    
    # Create all other visualizations with the combined approach
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS BY TIME PERIODS")
    print("=" * 60)
    (df_streams, df_listeners, 
     df_time_streams_combined, df_time_listeners_combined,
     df_ratios, df_momentum_time_combined, df_momentum_album, summary) = create_comprehensive_visualizations(results)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nKey insights:")
    print("✓ Average streams visualization shows performance across all runs with error bars")
    print("✓ Bar charts show average streams and listeners per album for each strategy")
    print("✓ Combined line charts show how total streaming and listener counts evolve over time")
    print("✓ Immediate release strategy extends with final values after its end period (shown by vertical dotted line)")
    print("✓ Dashed lines show Album 4 performance (zero fanbase album)")
    print("✓ Ratio chart shows repeat listening behavior (streams per unique listener)")
    print("✓ Momentum charts show algorithmic discovery patterns between similar users")
    print("✓ Summary table shows which strategy performs better overall for all metrics")
    print("✓ Gray dotted vertical line indicates where immediate release strategy ends")
    print("✓ Immediate release values after end period represent the plateau effect")
    
    return {
        'results': results,
        'average_streams_summary': average_streams_summary,
        'album_performance_summary': album_performance_summary,
        'df_streams': df_streams,
        'df_listeners': df_listeners,
        'df_time_streams_combined': df_time_streams_combined,
        'df_time_listeners_combined': df_time_listeners_combined,
        'df_ratios': df_ratios,
        'df_momentum_time_combined': df_momentum_time_combined,
        'df_momentum_album': df_momentum_album,
        'summary': summary
    }

def save_analysis_to_csv(analysis_results, output_dir="analysis_output"):
    """Save all analysis results to CSV files for further use"""
    print(f"\nSaving analysis results to '{output_dir}' directory...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save all DataFrames
    analysis_results['average_streams_summary'].to_csv(f"{output_dir}/average_streams_summary.csv", index=False)
    analysis_results['album_performance_summary'].to_csv(f"{output_dir}/album_performance_summary.csv", index=False)
    analysis_results['df_streams'].to_csv(f"{output_dir}/streams_by_album.csv", index=False)
    analysis_results['df_listeners'].to_csv(f"{output_dir}/listeners_by_album.csv", index=False)
    analysis_results['df_time_streams_combined'].to_csv(f"{output_dir}/time_streams_combined.csv", index=False)
    analysis_results['df_time_listeners_combined'].to_csv(f"{output_dir}/time_listeners_combined.csv", index=False)
    analysis_results['df_ratios'].to_csv(f"{output_dir}/streams_per_listener_ratios.csv", index=False)
    analysis_results['df_momentum_time_combined'].to_csv(f"{output_dir}/momentum_time_combined.csv", index=False)
    analysis_results['df_momentum_album'].to_csv(f"{output_dir}/momentum_by_album.csv", index=False)
    analysis_results['summary'].to_csv(f"{output_dir}/overall_summary.csv", index=False)
    
    print(f"✓ All analysis results saved to '{output_dir}' directory")

# Main execution functions
def run_full_analysis(results_dir="simulation_results", save_output=True, output_dir="analysis_output"):
    """Run the complete analysis pipeline"""
    print("🚀 Starting full analysis pipeline...")
    
    # Generate all visualizations
    analysis_results = generate_all_visualizations(results_dir)
    
    if analysis_results is None:
        return None

    
    # Save results if requested
    if save_output:
        save_analysis_to_csv(analysis_results, output_dir)
    
    print("✅ Full analysis pipeline completed successfully!")
    return analysis_results

def run_specific_visualization(results_dir="simulation_results", viz_type="all"):
    """Run specific visualization types"""
    print(f"🎯 Running specific visualization: {viz_type}")
    
    # Load data
    results = load_simulation_results(results_dir)
    if results is None:
        return None
    
    if viz_type == "average_streams" or viz_type == "all":
        print("\n📊 Creating Average Streams Visualization...")
        create_average_streams_visualization(results)
    
    if viz_type == "album_performance" or viz_type == "all":
        print("\n📈 Creating Album Performance Visualization...")
        create_album_performance_by_runs_visualization(results)
    
    if viz_type == "comprehensive" or viz_type == "all":
        print("\n🔍 Creating Comprehensive Time-Series Visualizations...")
        create_comprehensive_visualizations(results)
    
    
    return results

# Main execution
if __name__ == "__main__":
    print("🎵 Music Album Release Strategy - Visualization Generator")
    print("This script reads CSV files from simulation runs and creates visualizations")
    print()
    
    print("\n" + "=" * 50)
    print("RUNNING DEFAULT ANALYSIS")
    print("=" * 50)
    
    # Run the full analysis by default
    try:
        analysis_results = run_full_analysis()
        
        if analysis_results:
            print("\n✅ SUCCESS: All visualizations have been generated!")
            print("\n📁 Output files:")
            print("   • Visualizations: Displayed in matplotlib windows")
            print("   • Analysis CSVs: Saved to 'analysis_output/' directory")
            print("   • Original data: Located in 'simulation_results/' directory")
            print("\n🔄 To regenerate visualizations anytime, simply run this script again!")
        else:
            print("\n❌ FAILED: Could not load simulation data")
            print("Please ensure simulation CSV files exist in 'simulation_results/' directory")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check that:")
        print("1. Required libraries are installed (pandas, matplotlib, numpy)")
        print("2. CSV files exist in the expected directory")
        print("3. CSV files follow the expected naming pattern")
    
    print("\n" + "=" * 80)
    print("SCRIPT COMPLETED")
    print("=" * 80)
