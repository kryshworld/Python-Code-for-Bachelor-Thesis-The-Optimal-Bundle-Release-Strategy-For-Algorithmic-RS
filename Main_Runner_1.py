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
    from Main_Thesis_Simulation_Code import *  # or whatever you named it
    print("Thesis file imported successfully")
    print(f"Successfully imported I = {I}")
except Exception as e:
    print(f"Import error: {e}")
    exit()

print("All imports successful, continuing...")

def extract_album_streams(play_count, albums):
    """Extract total streams per album from play_count matrix"""
    album_streams = []
    for album_id in albums.keys():
        # Sum all plays for songs in this album
        album_total = np.sum(play_count[:, albums[album_id]])
        album_streams.append(album_total)
    return album_streams

def extract_album_listeners(R_mask, albums):
    """Extract unique listeners per album from R_mask matrix"""
    album_listeners = []
    for album_id in albums.keys():
        # Count unique users who have listened to at least one song from this album
        album_songs = albums[album_id]
        # Check if any user has listened to any song from this album
        users_listened = np.any(R_mask[:, album_songs], axis=1)
        unique_listeners = np.sum(users_listened)
        album_listeners.append(unique_listeners)
    return album_listeners

def extract_streams_over_time(recommendations_history, albums, I):
    """Extract cumulative streams over time periods"""
    streams_over_time = {album_id: [] for album_id in albums.keys()}
    cumulative_streams = {album_id: 0 for album_id in albums.keys()}
    
    for period, recommendations in enumerate(recommendations_history):
        # Count plays in this period for each album
        period_plays = {album_id: 0 for album_id in albums.keys()}
        
        for user_id, rec_song, actual_song, followed in recommendations:
            # Check which album the actual song belongs to
            for album_id, song_list in albums.items():
                if actual_song in song_list:
                    period_plays[album_id] += 1
                    break
        
        # Update cumulative streams
        for album_id in albums.keys():
            cumulative_streams[album_id] += period_plays[album_id]
            streams_over_time[album_id].append(cumulative_streams[album_id])
    
    return streams_over_time

def extract_listeners_over_time(recommendations_history, albums, I):
    """Extract cumulative unique listeners over time periods"""
    listeners_over_time = {album_id: [] for album_id in albums.keys()}
    # Track which users have listened to each album
    users_listened_to_album = {album_id: set() for album_id in albums.keys()}
    
    for period, recommendations in enumerate(recommendations_history):
        # Track new listeners in this period for each album
        for user_id, rec_song, actual_song, followed in recommendations:
            # Check which album the actual song belongs to
            for album_id, song_list in albums.items():
                if actual_song in song_list:
                    users_listened_to_album[album_id].add(user_id)
                    break
        
        # Update cumulative unique listeners count
        for album_id in albums.keys():
            listeners_over_time[album_id].append(len(users_listened_to_album[album_id]))
    
    return listeners_over_time

def calculate_momentum_metric(recommendations_history, albums, R_observed):
    """Calculate momentum metric by time period and album"""
    momentum_data = {album_id: [] for album_id in albums.keys()}
    
    # Track when each user discovers each song (first time listening)
    user_song_discovery = {}  # {user_id: {song_id: period}}
    
    for period, recommendations in enumerate(recommendations_history):
        # Track discoveries in this period
        period_discoveries = {album_id: [] for album_id in albums.keys()}
        
        for user_id, rec_song, actual_song, followed in recommendations:
            # Check if this is a new discovery (first time listening to this song)
            if user_id not in user_song_discovery:
                user_song_discovery[user_id] = {}
            
            if actual_song not in user_song_discovery[user_id]:
                user_song_discovery[user_id][actual_song] = period
                
                # Find which album this song belongs to
                for album_id, song_list in albums.items():
                    if actual_song in song_list:
                        period_discoveries[album_id].append(user_id)
                        break
        
        # Calculate momentum for each album in this period
        for album_id in albums.keys():
            momentum_count = 0
            discoverers = period_discoveries[album_id]
            
            # Check all pairs of users who discovered songs from this album
            for i in range(len(discoverers)):
                for j in range(i + 1, len(discoverers)):
                    user1, user2 = discoverers[i], discoverers[j]
                    
                    # Calculate user similarity based on shared ratings
                    similarity = calculate_user_similarity(user1, user2, R_observed)
                    
                    if similarity > 0.7:  # Strong similarity threshold
                        # Check if they discovered within consecutive periods
                        songs_user1 = [song for song, disc_period in user_song_discovery[user1].items() 
                                     if song in albums[album_id] and abs(disc_period - period) <= 1]
                        songs_user2 = [song for song, disc_period in user_song_discovery[user2].items() 
                                     if song in albums[album_id] and abs(disc_period - period) <= 1]
                        
                        # Check for shared discoveries within 2 consecutive periods
                        shared_recent = set(songs_user1) & set(songs_user2)
                        if shared_recent:
                            momentum_count += len(shared_recent)
            
            momentum_data[album_id].append(momentum_count)
    
    return momentum_data

def calculate_user_similarity(user1, user2, R_observed):
    """Calculate Pearson correlation between two users based on shared song ratings"""
    # Get songs both users have rated (non-zero entries)
    user1_ratings = R_observed[user1, :]
    user2_ratings = R_observed[user2, :]
    
    # Find songs both users have rated
    both_rated = (user1_ratings != 0) & (user2_ratings != 0)
    
    if np.sum(both_rated) < 3:  # Need at least 3 shared ratings
        return 0
    
    shared_ratings_user1 = user1_ratings[both_rated]
    shared_ratings_user2 = user2_ratings[both_rated]
    
    # Calculate means
    mean1 = np.mean(shared_ratings_user1)
    mean2 = np.mean(shared_ratings_user2)
    
    # Calculate standard deviations
    std1 = np.std(shared_ratings_user1, ddof=0)
    std2 = np.std(shared_ratings_user2, ddof=0)
    
    # Handle cases where standard deviation is zero (constant ratings)
    if std1 == 0 or std2 == 0:
        # If both users have constant ratings and they're the same, perfect correlation
        if std1 == 0 and std2 == 0:
            return 1.0 if mean1 == mean2 else 0.0
        # If only one user has constant ratings, no correlation
        else:
            return 0.0
    
    # Calculate Pearson correlation manually
    numerator = np.mean((shared_ratings_user1 - mean1) * (shared_ratings_user2 - mean2))
    correlation = numerator / (std1 * std2)
    
    return correlation

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
    results_dir = "Analysis_Results22"
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
    results_dir = "simulation_results22"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create filename with run number and seed
    base_filename = f"run_{run_number}_seed_{run_seed}"

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

    # 1. Album-level results
    album_data = {
        'Album_ID': list(range(len(run_data['gradual_album_streams']))),
        'Gradual_Album_Streams': run_data['gradual_album_streams'],
        'Immediate_Album_Streams': run_data['immediate_album_streams'],
        'Gradual_Album_Listeners': run_data['gradual_album_listeners'],
        'Immediate_Album_Listeners': run_data['immediate_album_listeners']
    }
    
    album_df = pd.DataFrame(album_data)
    album_df.to_csv(f"{results_dir}/{base_filename}_album_results.csv", index=False)
    
    # 2. Time series data - streams
    max_periods = max(
        max(len(streams) for streams in run_data['gradual_time_streams'].values()),
        max(len(streams) for streams in run_data['immediate_time_streams'].values())
    )
    
    time_streams_data = {'Time_Period': list(range(1, max_periods + 1))}
    
    # Add gradual streams for each album
    for album_id in run_data['gradual_time_streams'].keys():
        gradual_streams = run_data['gradual_time_streams'][album_id]
        # Pad with last value if shorter
        while len(gradual_streams) < max_periods:
            gradual_streams.append(gradual_streams[-1] if gradual_streams else 0)
        time_streams_data[f'Gradual_Album_{album_id}_Streams'] = gradual_streams
    
    # Add immediate streams for each album
    for album_id in run_data['immediate_time_streams'].keys():
        immediate_streams = run_data['immediate_time_streams'][album_id]
        # Pad with last value if shorter
        while len(immediate_streams) < max_periods:
            immediate_streams.append(immediate_streams[-1] if immediate_streams else 0)
        time_streams_data[f'Immediate_Album_{album_id}_Streams'] = immediate_streams
    
    time_streams_df = pd.DataFrame(time_streams_data)
    time_streams_df.to_csv(f"{results_dir}/{base_filename}_time_streams.csv", index=False)
    
    # 3. Time series data - listeners
    time_listeners_data = {'Time_Period': list(range(1, max_periods + 1))}
    
    # Add gradual listeners for each album
    for album_id in run_data['gradual_time_listeners'].keys():
        gradual_listeners = run_data['gradual_time_listeners'][album_id]
        # Pad with last value if shorter
        while len(gradual_listeners) < max_periods:
            gradual_listeners.append(gradual_listeners[-1] if gradual_listeners else 0)
        time_listeners_data[f'Gradual_Album_{album_id}_Listeners'] = gradual_listeners
    
    # Add immediate listeners for each album
    for album_id in run_data['immediate_time_listeners'].keys():
        immediate_listeners = run_data['immediate_time_listeners'][album_id]
        # Pad with last value if shorter
        while len(immediate_listeners) < max_periods:
            immediate_listeners.append(immediate_listeners[-1] if immediate_listeners else 0)
        time_listeners_data[f'Immediate_Album_{album_id}_Listeners'] = immediate_listeners
    
    time_listeners_df = pd.DataFrame(time_listeners_data)
    time_listeners_df.to_csv(f"{results_dir}/{base_filename}_time_listeners.csv", index=False)
    
    # 4. Momentum data
    momentum_data = {'Time_Period': list(range(1, max_periods + 1))}
    
    # Add gradual momentum for each album
    for album_id in run_data['gradual_momentum'].keys():
        gradual_momentum = run_data['gradual_momentum'][album_id]
        # Pad with zeros if shorter
        while len(gradual_momentum) < max_periods:
            gradual_momentum.append(0)
        momentum_data[f'Gradual_Album_{album_id}_Momentum'] = gradual_momentum
    
    # Add immediate momentum for each album
    for album_id in run_data['immediate_momentum'].keys():
        immediate_momentum = run_data['immediate_momentum'][album_id]
        # Pad with zeros if shorter
        while len(immediate_momentum) < max_periods:
            immediate_momentum.append(0)
        momentum_data[f'Immediate_Album_{album_id}_Momentum'] = immediate_momentum
    
    momentum_df = pd.DataFrame(momentum_data)
    momentum_df.to_csv(f"{results_dir}/{base_filename}_momentum.csv", index=False)
    
    print(f"Saved results for Run {run_number} (seed {run_seed}) to CSV files in '{results_dir}' directory")

def run_multiple_simulations(num_runs):
    """Run the simulation multiple times and collect results"""
    all_results = []
    
    for run in range(num_runs):
        print(f"Running simulation {run + 1}/{num_runs}")
        
        # Set seed for reproducibility
        run_seed = start_seed + run
        print(f"Using seed: {run_seed}")
        
        # Generate fresh user properties for this run
        user_properties, fan_properties, recommendation_adherence = generate_user_properties(I, seed=run_seed)

        R_truth,  _, _ = initialize_truth_matrix(I, J, H)

        # Run the initial market simulation
        T_hat_1, V_hat_1, R_observed_1, R_mask_1, _, _ = music_market(R_truth, user_properties, seed=run_seed)
        
        # Create fresh album fanbases for this run
        album_fanbases_run = create_album_fanbases(albums, I, seed=run_seed)
        
        # Get Starting Positions
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
        
        # Extract results
        gradual_album_streams = extract_album_streams(play_count_grad, albums)
        immediate_album_streams = extract_album_streams(play_count_imm, albums)
        
        gradual_album_listeners = extract_album_listeners(R_mask_grad, albums)
        immediate_album_listeners = extract_album_listeners(R_mask_imm, albums)
        
        gradual_time_streams = extract_streams_over_time(recommendations_history_grad, albums, I)
        immediate_time_streams = extract_streams_over_time(recommendations_history_imm, albums, I)
        
        gradual_time_listeners = extract_listeners_over_time(recommendations_history_grad, albums, I)
        immediate_time_listeners = extract_listeners_over_time(recommendations_history_imm, albums, I)
        
        # Calculate momentum metrics
        gradual_momentum = calculate_momentum_metric(recommendations_history_grad, albums, R_observed_grad)
        immediate_momentum = calculate_momentum_metric(recommendations_history_imm, albums, R_observed_imm)
        
        run_data = {
            'run': run + 1,
            'seed': run_seed,
            'gradual_album_streams': gradual_album_streams,
            'immediate_album_streams': immediate_album_streams,
            'gradual_album_listeners': gradual_album_listeners,
            'immediate_album_listeners': immediate_album_listeners,
            'gradual_time_streams': gradual_time_streams,
            'immediate_time_streams': immediate_time_streams,
            'gradual_time_listeners': gradual_time_listeners,
            'immediate_time_listeners': immediate_time_listeners,
            'gradual_momentum': gradual_momentum,
            'immediate_momentum': immediate_momentum,
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
    start_seed =2907
    results = run_multiple_simulations(6)
    
    print("\nAll analysis and exports completed!")
