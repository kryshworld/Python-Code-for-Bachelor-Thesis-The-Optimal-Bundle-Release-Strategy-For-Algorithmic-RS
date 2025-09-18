print("Starting file...")

import pandas as pd
print("Pandas imported successfully")

import matplotlib.pyplot as plt
print("Matplotlib imported successfully")

import numpy as np
print("Numpy imported successfully")

import os
print("OS imported successfully")

# Try the import from your thesis file
try:
    from Main_Thesis19 import *  # or whatever you named it
    print("Thesis file imported successfully")
    print(f"Successfully imported I = {I}")
except Exception as e:
    print(f"Import error: {e}")
    exit()

print("All imports successful, continuing...")

def generate_user_properties(I, seed=None):
    """Generate user properties for a simulation run"""
    if seed is not None:
        np.random.seed(seed)
    
    user_properties = {
        'explorer_score': np.random.beta(1, 22, size=I)
    }
    
    fan_properties = {
        'fan_bias': np.random.beta(1, 22, size=I)
    }
    
    recommendation_adherence = np.random.beta(2, 8, size=I)
    
    return user_properties, fan_properties, recommendation_adherence

def save_time_period_results_to_csv(T_hat_history, V_hat_history, run_number, run_seed, strategy_name):
    """Save T_hat and V_hat for each time period to CSV files"""
    
    # Create directory if it doesn't exist
    results_dir = "Analysis_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create filename with run number, seed, and strategy
    base_filename = f"run_{run_number}_seed_{run_seed}_{strategy_name}"
    
    # Save each time period's matrices
    for t, (T_hat, V_hat) in enumerate(zip(T_hat_history, V_hat_history)):
        # Save T_hat for this time period
        T_hat_df = pd.DataFrame(T_hat)
        T_hat_df.to_csv(f"{results_dir}/{base_filename}_T_hat_period_{t+1}.csv", index=False)
        
        # Save V_hat for this time period
        V_hat_df = pd.DataFrame(V_hat)
        V_hat_df.to_csv(f"{results_dir}/{base_filename}_V_hat_period_{t+1}.csv", index=False)
    
    print(f"Saved {strategy_name} time period results for Run {run_number} (seed {run_seed}) - {len(T_hat_history)} time periods")

def save_run_results_to_csv(run_data, run_number, run_seed):
    """Save all results from a single run to CSV files"""
    
    # Create directory if it doesn't exist
    results_dir = "Analysis_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create filename with run number and seed
    base_filename = f"run_{run_number}_seed_{run_seed}"
    
    T_hat_start_df = pd.DataFrame(run_data['T_hat_start'])
    T_hat_start_df.to_csv(f"{results_dir}/{base_filename}_T_hat_start.csv", index=False)

    V_hat_start_df = pd.DataFrame(run_data['V_hat_start'])
    V_hat_start_df.to_csv(f"{results_dir}/{base_filename}_V_hat_start.csv", index=False)

    # Save final T hat and V hat to CSV files (keeping your original functionality)
    T_hat__grad_df = pd.DataFrame(run_data['T_hats_grad'])
    T_hat__grad_df.to_csv(f"{results_dir}/{base_filename}_T_hat_grad_final.csv", index=False)

    V_hat__grad_df = pd.DataFrame(run_data['V_hats_grad'])
    V_hat__grad_df.to_csv(f"{results_dir}/{base_filename}_V_hat_grad_final.csv", index=False)

    T_hat__imm_df = pd.DataFrame(run_data['T_hats_imm'])
    T_hat__imm_df.to_csv(f"{results_dir}/{base_filename}_T_hat_imm_final.csv", index=False)

    V_hat__imm_df = pd.DataFrame(run_data['V_hats_imm'])
    V_hat__imm_df.to_csv(f"{results_dir}/{base_filename}_V_hat_imm_final.csv", index=False)
    
    # Save time period results if they exist
    if 'T_hat_history_start' in run_data and 'V_hat_history_start' in run_data:
        save_time_period_results_to_csv(
            run_data['T_hat_history_start'], 
            run_data['V_hat_history_start'], 
            run_number, 
            run_seed, 
            "starting_positions"
        )


    if 'T_hat_history_grad' in run_data and 'V_hat_history_grad' in run_data:
        save_time_period_results_to_csv(
            run_data['T_hat_history_grad'], 
            run_data['V_hat_history_grad'], 
            run_number, 
            run_seed, 
            "gradual"
        )
    
    if 'T_hat_history_imm' in run_data and 'V_hat_history_imm' in run_data:
        save_time_period_results_to_csv(
            run_data['T_hat_history_imm'], 
            run_data['V_hat_history_imm'], 
            run_number, 
            run_seed, 
            "immediate"
        )

    print(f"Saved results for Run {run_number} (seed {run_seed}) to CSV files in '{results_dir}' directory")

def run_multiple_simulations(num_runs):
    """Run the simulation multiple times and collect results"""
    all_results = []
    
    for run in range(num_runs):
        print(f"Running simulation {run + 1}/{num_runs}")
        
        # Set seed for reproducibility
        run_seed = 1684 + run
        print(f"Using seed: {run_seed}")
        
        # Generate fresh user properties for this run
        user_properties, fan_properties, recommendation_adherence = generate_user_properties(I, seed=run_seed)

        R_truth,  _, _ = initialize_truth_matrix(I, J, H)

        # Run the initial market simulation
        T_hat_1, V_hat_1, R_observed_1, R_mask_1, _, _ = music_market(R_truth, user_properties, seed=run_seed)
        
        # Create fresh album fanbases for this run
        album_fanbases_run = create_album_fanbases(albums, I, seed=run_seed)

        # Run starting positions
        (R_truth_start, T_hat_start, V_hat_start, R_observed_start, 
         R_mask_start, recommendations_history_start, play_count_start, _, 
         T_hat_history_start, V_hat_history_start) = starting_positions(
            album_fanbases_run, albums, J_new, J, I, 
            T_hat_1, V_hat_1, R_observed_1, R_mask_1, R_truth, user_properties, fan_properties, recommendation_adherence, seed=run_seed
        )

        
        # Run gradual release strategy
        (R_truth_grad, T_hat_grad, V_hat_grad, R_observed_grad, 
         R_mask_grad, recommendations_history_grad, play_count_grad, _, 
         T_hat_history_grad, V_hat_history_grad) = album_release_gradual(
            album_fanbases_run, albums, J_new, J, I, 
            T_hat_1, V_hat_1, R_observed_1, R_mask_1, R_truth, user_properties, fan_properties, recommendation_adherence, seed=run_seed
        )
        
        # Run immediate release strategy (reset to initial state)
        (R_truth_imm, T_hat_imm, V_hat_imm, R_observed_imm, 
         R_mask_imm, recommendations_history_imm, play_count_imm, _, 
         T_hat_history_imm, V_hat_history_imm) = album_release_immediate(
            album_fanbases_run, albums, J_new, J, I, 
            T_hat_1, V_hat_1, R_observed_1, R_mask_1, R_truth, user_properties, fan_properties, recommendation_adherence, seed=run_seed
        )
        
        run_data = {
            'run': run + 1,
            'seed': run_seed,
            'T_hat_start': T_hat_start,
            'V_hat_start': V_hat_start,
            'T_hat_history_start': T_hat_history_start,
            'V_hat_history_start': V_hat_history_start,
            'T_hats_grad': T_hat_grad,
            'V_hats_grad': V_hat_grad,
            'T_hats_imm': T_hat_imm,
            'V_hats_imm': V_hat_imm,
            'T_hat_history_grad': T_hat_history_grad,
            'V_hat_history_grad': V_hat_history_grad,
            'T_hat_history_imm': T_hat_history_imm,
            'V_hat_history_imm': V_hat_history_imm,
        }
        
        # Save this run's results to CSV
        save_run_results_to_csv(run_data, run + 1, run_seed)
        
        all_results.append(run_data)
    
    return all_results

# Main execution
if __name__ == "__main__":
    print("Starting analysis with imported simulation functions...")
    print("=" * 60)
    
    # Print some info about imported variables
    print(f"Number of users (I): {I}")
    print(f"Original songs (J): {J}")
    print(f"Total songs with albums (J_new): {J_new}")
    print(f"Number of albums: {num_of_albums}")
    print(f"Songs per album: {album_len}")
    print()
    
    print("Albums structure:")
    for album_id, songs in albums.items():
        print(f"  Album {album_id}: songs {songs}")
    print()
    
    # Run multiple simulations
    print("Running simulations...")
    results = run_multiple_simulations(1)
    
    # Create directory if it doesn't exist
    results_dir = "Analysis_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print("\nAll analysis and exports completed!")
