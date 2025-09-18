import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # ADD THIS IMPORT
import numpy as np
import pandas as pd
import os


results_dir = "Analysis_Results"
run_seed = 1670  # Use the current seed in the loop
run_number = 7

def load_matrix_period_data(run_number, run_seed, strategy, matrix_type, period):
    """Load T_hat or V_hat matrix for a specific time period"""
    filename = f"run_{run_number}_seed_{run_seed}_{strategy}_{matrix_type}_period_{period}.csv"
    filepath = os.path.join(results_dir, filename)
    
    if os.path.exists(filepath):
        data = pd.read_csv(filepath, header=None)
        return data.values.astype(float)
    else:
        print(f"Warning: File not found: {filename}")
        return None

def plot_hypersphere_evolution(run_number=1):
    """Create hypersphere plots showing T_hat and V_hat evolution over time periods"""
    
    # Check if Analysis_Results directory exists
    if not os.path.exists(results_dir):
        print("Error: Analysis_Results directory not found!")
        print("Please run your modified Analysis_Runner.py first to generate the matrix files.")
        return
    
    # Create starting positions plot (single period)
    print("Creating Starting Positions Hypersphere (Period 1 only)...")

    # Create single figure for starting positions
    fig_start = plt.figure(figsize=(8, 6))  # Smaller figure for single plot
    ax_start = fig_start.add_subplot(111)

    # Generate hypersphere (unit circle) in the positive orthant
    theta = np.linspace(0, np.pi/2, 100, endpoint=True)
    sphere_x = np.cos(theta)
    sphere_y = np.sin(theta)

    # Load T_hat and V_hat for period 1 only
    t_hat_matrix = load_matrix_period_data(run_number, run_seed, "starting_positions", "T_hat", 1)
    v_hat_matrix = load_matrix_period_data(run_number, run_seed, "starting_positions", "V_hat", 1)

    if t_hat_matrix is not None and v_hat_matrix is not None:
        # Extract T_hat values (assuming 2D latent factors)
        t_hat_x = t_hat_matrix[:, 0]
        t_hat_y = t_hat_matrix[:, 1]
        
        # Extract V_hat values
        v_hat_x = v_hat_matrix[:, 0]
        v_hat_y = v_hat_matrix[:, 1]
        
        # Separate V_hat into different categories
        # Main songs (first 801 songs)
        v_hat_main_x = v_hat_x[:801] if len(v_hat_x) >= 801 else v_hat_x
        v_hat_main_y = v_hat_y[:801] if len(v_hat_y) >= 801 else v_hat_y
        
        # New album songs (last 6 songs)
        if len(v_hat_x) > 801:
            # Album songs (songs 801-806)
            v_hat_album_x = v_hat_x[801:807] if len(v_hat_x) >= 807 else v_hat_x[801:]
            v_hat_album_y = v_hat_y[801:807] if len(v_hat_y) >= 807 else v_hat_y[801:]
            
            # Split album songs into two albums if we have 6 songs
            if len(v_hat_album_x) == 6:
                v_hat_red_x = v_hat_album_x[:3]  # First album (Album 0)
                v_hat_red_y = v_hat_album_y[:3]
                v_hat_green_x = v_hat_album_x[3:]  # Second album (Album 1)
                v_hat_green_y = v_hat_album_y[3:]
            else:
                v_hat_red_x = v_hat_album_x
                v_hat_red_y = v_hat_album_y
                v_hat_green_x = np.array([])
                v_hat_green_y = np.array([])
        else:
            v_hat_red_x = np.array([])
            v_hat_red_y = np.array([])
            v_hat_green_x = np.array([])
            v_hat_green_y = np.array([])
        
        # Plot the hypersphere
        ax_start.plot(sphere_x, sphere_y, 'r-', linewidth=3, alpha=1, label='Hypersphere')
        
        # Plot T_hat values as orange dots
        ax_start.scatter(t_hat_x, t_hat_y, c='orange', s=2, alpha=0.3, edgecolors='none', label='T_hat values')
        
        # Plot main V_hat values as blue
        ax_start.scatter(v_hat_main_x, v_hat_main_y, c='blue', s=8, alpha=0.6, edgecolors='none', label='V_hat (original songs)')
        
        # Plot album songs if they exist
        if len(v_hat_red_x) > 0:
            ax_start.scatter(v_hat_red_x, v_hat_red_y, c='red', s=110, alpha=0.9, 
                        edgecolors='black', linewidths=1, label='Album 0 songs')
        
        if len(v_hat_green_x) > 0:
            ax_start.scatter(v_hat_green_x, v_hat_green_y, c='green', s=110, alpha=0.9, 
                        edgecolors='black', linewidths=1, label='Album 1 songs')
        
        # Set up the plot with explicit font sizes
        ax_start.set_xlim(-0.1, 1.1)
        ax_start.set_ylim(-0.1, 1.1)
        ax_start.set_xlabel('h1', fontsize=20)
        ax_start.set_ylabel('h2', fontsize=20)
        ax_start.set_xticks([0, 1.0])
        ax_start.set_yticks([0, 1.0])
        ax_start.grid(True, alpha=0.3)
        for grid_val in [0.25, 0.5, 0.75]:
            ax_start.axhline(y=grid_val, color='gray', linewidth=0.5, alpha=0.3)
            ax_start.axvline(x=grid_val, color='gray', linewidth=0.5, alpha=0.3)
        ax_start.set_aspect('equal')
        ax_start.tick_params(axis='both', which='major', labelsize=15)
        
        # Add axis lines
        ax_start.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax_start.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        
        # Print statistics
        print(f"Starting Positions - T_hat points: {len(t_hat_x)}, V_hat points: {len(v_hat_x)}")

    plt.tight_layout()
    plt.savefig(f'Starting_Positions_{run_seed}.png', dpi=300, bbox_inches='tight')

    # Create gradual strategy plots (8 periods)
    print("Creating Gradual Strategy Hypersphere Evolution (8 periods)...")
    
    # WITH THIS:
    fig_grad = plt.figure(figsize=(21, 13))
    gs_grad = gridspec.GridSpec(2, 4, figure=fig_grad)
    axes_grad = []
    for i in range(8):
        ax = fig_grad.add_subplot(gs_grad[i//4, i%4])
        axes_grad.append(ax)
    
    # Generate hypersphere (unit circle) in the positive orthant
    theta = np.linspace(0, np.pi/2, 100, endpoint=True)
    sphere_x = np.cos(theta)
    sphere_y = np.sin(theta)

    for period in range(1, 9):  # 8 periods for gradual
        ax = axes_grad[period - 1]
        
        # Load T_hat and V_hat for this period
        t_hat_matrix = load_matrix_period_data(run_number, run_seed, "gradual", "T_hat", period)
        v_hat_matrix = load_matrix_period_data(run_number, run_seed, "gradual", "V_hat", period)
        
        if t_hat_matrix is not None and v_hat_matrix is not None:
            # Extract T_hat values (assuming 2D latent factors)
            t_hat_x = t_hat_matrix[:, 0]
            t_hat_y = t_hat_matrix[:, 1]
            
            # Extract V_hat values
            v_hat_x = v_hat_matrix[:, 0]
            v_hat_y = v_hat_matrix[:, 1]
            
            # Separate V_hat into different categories
            # Main songs (first 801 songs)
            v_hat_main_x = v_hat_x[:801] if len(v_hat_x) >= 801 else v_hat_x
            v_hat_main_y = v_hat_y[:801] if len(v_hat_y) >= 801 else v_hat_y
            
            # New album songs (last 6 songs)
            if len(v_hat_x) > 801:
                # Album songs (songs 801-806)
                v_hat_album_x = v_hat_x[801:807] if len(v_hat_x) >= 807 else v_hat_x[801:]
                v_hat_album_y = v_hat_y[801:807] if len(v_hat_y) >= 807 else v_hat_y[801:]
                
                # Split album songs into two albums if we have 6 songs
                if len(v_hat_album_x) == 6:
                    v_hat_red_x = v_hat_album_x[:3]  # First album (Album 0)
                    v_hat_red_y = v_hat_album_y[:3]
                    v_hat_green_x = v_hat_album_x[3:]  # Second album (Album 1)
                    v_hat_green_y = v_hat_album_y[3:]
                else:
                    v_hat_red_x = v_hat_album_x
                    v_hat_red_y = v_hat_album_y
                    v_hat_green_x = np.array([])
                    v_hat_green_y = np.array([])
            else:
                v_hat_red_x = np.array([])
                v_hat_red_y = np.array([])
                v_hat_green_x = np.array([])
                v_hat_green_y = np.array([])
            
            # Plot the hypersphere
            ax.plot(sphere_x, sphere_y, 'r-', linewidth=3, alpha=1, label='Hypersphere')
            
            # Plot T_hat values as orange dots
            ax.scatter(t_hat_x, t_hat_y, c='orange', s=2, alpha=0.3, edgecolors='none', label='T_hat values')
            
            # Plot main V_hat values as blue
            ax.scatter(v_hat_main_x, v_hat_main_y, c='blue', s=8, alpha=0.6, edgecolors='none', label='V_hat (original songs)')
            
            # Plot album songs if they exist
            if len(v_hat_red_x) > 0:
                ax.scatter(v_hat_red_x, v_hat_red_y, c='red', s=110, alpha=0.9, 
                        edgecolors='black', linewidths=1, label='Album 0 songs')
            
            if len(v_hat_green_x) > 0:
                ax.scatter(v_hat_green_x, v_hat_green_y, c='green', s=110, alpha=0.9, 
                        edgecolors='black', linewidths=1, label='Album 1 songs')
            
            # Set up the plot with explicit font sizes
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel('h1', fontsize=20)
            ax.set_ylabel('h2', fontsize=20)
            ax.set_title(f'Period {period}', fontweight='bold', fontsize=27, y=1.06)
            ax.set_xticks([0, 1.0])
            ax.set_yticks([0, 1.0])  # Adjust y for better title placement
            ax.grid(True, alpha=0.3)
            for grid_val in [0.25, 0.5, 0.75]:
                ax.axhline(y=grid_val, color='gray', linewidth=0.5, alpha=0.3)
                ax.axvline(x=grid_val, color='gray', linewidth=0.5, alpha=0.3)
            ax.set_aspect('equal')
            ax.tick_params(axis='both', which='major', labelsize=15)
            
            # Add axis lines
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
            
            # Print statistics for this period
            print(f"Period {period} - T_hat points: {len(t_hat_x)}, V_hat points: {len(v_hat_x)}")
    
    plt.tight_layout()
    plt.savefig(f'Gradual_8_periods{run_seed}.png', dpi=300, bbox_inches='tight')
    
    # Create immediate strategy plots (6 periods)
    print("\nCreating Immediate Strategy Hypersphere Evolution (6 periods)...")
    
    # WITH THIS:
    fig_imm = plt.figure(figsize=(21, 13))  # Same figure size as gradual
    gs_imm = gridspec.GridSpec(2, 4, figure=fig_imm)  # Same grid as gradual
    axes_imm = []
    
    # Custom positioning: periods 1-4 in top row, periods 5-6 centered in bottom row
    positions = [
        (0, 0),  # Period 1: top row, col 0
        (0, 1),  # Period 2: top row, col 1
        (0, 2),  # Period 3: top row, col 2
        (0, 3),  # Period 4: top row, col 3
        (1, 1),  # Period 5: bottom row, col 1 (under period 2)
        (1, 2),  # Period 6: bottom row, col 2 (under period 3)
    ]
    
    for i in range(6):  # Only create 6 subplots for immediate
        row, col = positions[i]
        ax = fig_imm.add_subplot(gs_imm[row, col])
        axes_imm.append(ax)
    
    for period in range(1, 7):  # 6 periods for immediate
        ax = axes_imm[period - 1]
        
        # Load T_hat and V_hat for this period
        t_hat_matrix = load_matrix_period_data(run_number, run_seed, "immediate", "T_hat", period)
        v_hat_matrix = load_matrix_period_data(run_number, run_seed, "immediate", "V_hat", period)
        
        if t_hat_matrix is not None and v_hat_matrix is not None:
            # Extract T_hat values (assuming 2D latent factors)
            t_hat_x = t_hat_matrix[:, 0]
            t_hat_y = t_hat_matrix[:, 1]
            
            # Extract V_hat values
            v_hat_x = v_hat_matrix[:, 0]
            v_hat_y = v_hat_matrix[:, 1]
            
            # Separate V_hat into different categories
            # Main songs (first 801 songs)
            v_hat_main_x = v_hat_x[:801] if len(v_hat_x) >= 801 else v_hat_x
            v_hat_main_y = v_hat_y[:801] if len(v_hat_y) >= 801 else v_hat_y
            
            # New album songs (last 6 songs)
            if len(v_hat_x) > 801:
                # Album songs (songs 801-806)
                v_hat_album_x = v_hat_x[801:807] if len(v_hat_x) >= 807 else v_hat_x[801:]
                v_hat_album_y = v_hat_y[801:807] if len(v_hat_y) >= 807 else v_hat_y[801:]
                
                # Split album songs into two albums if we have 6 songs
                if len(v_hat_album_x) == 6:
                    v_hat_red_x = v_hat_album_x[:3]  # First album (Album 0)
                    v_hat_red_y = v_hat_album_y[:3]
                    v_hat_green_x = v_hat_album_x[3:]  # Second album (Album 1)
                    v_hat_green_y = v_hat_album_y[3:]
                else:
                    v_hat_red_x = v_hat_album_x
                    v_hat_red_y = v_hat_album_y
                    v_hat_green_x = np.array([])
                    v_hat_green_y = np.array([])
            else:
                v_hat_red_x = np.array([])
                v_hat_red_y = np.array([])
                v_hat_green_x = np.array([])
                v_hat_green_y = np.array([])
            
            # Plot the hypersphere
            ax.plot(sphere_x, sphere_y, 'r-', linewidth=3, alpha=1, label='Hypersphere')
            
            # Plot T_hat values as orange dots
            ax.scatter(t_hat_x, t_hat_y, c='orange', s=2, alpha=0.3, edgecolors='none', label='T_hat values')
            
            # Plot main V_hat values as blue
            ax.scatter(v_hat_main_x, v_hat_main_y, c='blue', s=8, alpha=0.6, edgecolors='none', label='V_hat (original songs)')
            
            # Plot album songs if they exist
            if len(v_hat_red_x) > 0:
                ax.scatter(v_hat_red_x, v_hat_red_y, c='red', s=110, alpha=0.9, 
                        edgecolors='black', linewidths=1, label='Album 0 songs')
            
            if len(v_hat_green_x) > 0:
                ax.scatter(v_hat_green_x, v_hat_green_y, c='green', s=110, alpha=0.9, 
                        edgecolors='black', linewidths=1, label='Album 1 songs')
            
            # Set up the plot with explicit font sizes
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel('h1', fontsize=20)
            ax.set_ylabel('h2', fontsize=20)
            ax.set_title(f'Period {period}', fontweight='bold', fontsize=27, y=1.06)
            ax.set_xticks([0, 1.0])
            ax.set_yticks([0, 1.0])  # Adjust y for better title placement
            ax.grid(True, alpha=0.3)
            for grid_val in [0.25, 0.5, 0.75]:
                ax.axhline(y=grid_val, color='gray', linewidth=0.5, alpha=0.3)
                ax.axvline(x=grid_val, color='gray', linewidth=0.5, alpha=0.3)
            ax.set_aspect('equal')
            ax.tick_params(axis='both', which='major', labelsize=15)
            
            # Add axis lines
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
            
            # Print statistics for this period
            print(f"Period {period} - T_hat points: {len(t_hat_x)}, V_hat points: {len(v_hat_x)}")
    
    plt.tight_layout()
    plt.savefig(f'Immediate_6_periods{run_seed}.png', dpi=300, bbox_inches='tight')

plot_hypersphere_evolution(run_number)
