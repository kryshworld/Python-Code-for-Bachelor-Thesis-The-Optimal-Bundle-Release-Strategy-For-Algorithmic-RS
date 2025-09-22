import numpy as np

# Parameters
I = 24030 # Number of users
J = 801  # Number of items
H = 2 # Latent factors
lambda_t = 0.01  # Regularization for T
lambda_v = 0.01 # Regularization for V
learning_rate = 0.05  # Learning rate for SGD
iterations = 50  # SGD iterations for each step
target_density = 0.01  # Continue until this density is reached
album_len = 3  # Number of songs per album
num_of_albums = 2  # Number of albums to simulate
rating_boost = 0.3  # Boost for songs rated above the mean

"""
Top 5 results by MSE:
1. λt=0.01, λv=0.01, lr=0.05 -> MSE=60.206926
2. λt=0.05, λv=0.01, lr=0.15 -> MSE=218.999495
3. λt=0.05, λv=0.01, lr=0.10 -> MSE=224.011092
4. λt=0.01, λv=0.05, lr=0.15 -> MSE=229.731417
5. λt=0.01, λv=0.05, lr=0.10 -> MSE=266.906436
"""

"""for half size w only 0.01
Top 5 results by MSE:
1. λt=0.01, λv=0.01, lr=0.20 -> MSE=238.824514
2. λt=0.05, λv=0.01, lr=0.15 -> MSE=248.011201
3. λt=0.10, λv=0.01, lr=0.20 -> MSE=251.317872
4. λt=0.10, λv=0.01, lr=0.10 -> MSE=252.147545
5. λt=0.01, λv=0.01, lr=0.15 -> MSE=255.730637"""

"""
1. λt=0.05, λv=0.01, lr=0.10 -> MSE=0.007118
2. λt=0.15, λv=0.01, lr=0.10 -> MSE=0.007703
3. λt=0.20, λv=0.01, lr=0.20 -> MSE=0.007737
4. λt=0.10, λv=0.01, lr=0.15 -> MSE=0.007794
5. λt=0.05, λv=0.01, lr=0.15 -> MSE=0.008148
"""

"""
1. λt=0.10, λv=0.01, lr=0.20 -> MSE=101.787507
2. λt=0.15, λv=0.01, lr=0.15 -> MSE=105.526409
3. λt=0.20, λv=0.01, lr=0.15 -> MSE=113.326175
4. λt=0.15, λv=0.01, lr=0.20 -> MSE=126.118585
5. λt=0.10, λv=0.01, lr=0.15 -> MSE=130.995319
"""

# Initialize truth matrix
def initialize_truth_matrix(I, J, H):
    T_truth = np.random.rand(I, H)
    T_truth = T_truth / np.linalg.norm(T_truth, axis=1, keepdims=True)  # Normalize each row

    V_truth = np.random.rand(J, H)
    V_truth = V_truth / np.linalg.norm(V_truth, axis=1, keepdims=True)  # Normalize each row

    # Compute matrix product
    U = T_truth @ V_truth.T  # Shape (I, J)
    
    # Calculate u̅ij (mean utility across users for each product j)
    u_bar = np.mean(U, axis=0)  # Shape (J,) - mean across users for each product
    
    # Calculate standard deviation of u̅ij across products
    u_bar_std = np.std(u_bar)
    
    # Set idiosyncratic noise to 10% of this standard deviation
    idiosyncratic_noise = (0.10 * u_bar_std) ** 2  # Variance = (10% of std)^2
    
    print(f"Mean utility std across products: {u_bar_std:.4f}")
    print(f"Idiosyncratic noise variance: {idiosyncratic_noise:.6f}")

    # Add Gaussian noise
    noise_U = np.random.normal(loc=0.0, scale=np.sqrt(idiosyncratic_noise), size=(I, J))
    U_noisy = U + noise_U

    tilde_noise = np.random.normal(loc=0.0, scale=np.sqrt(idiosyncratic_noise), size=(I, J))
    R_truth_not_normalized = U_noisy + tilde_noise

    R_truth = (R_truth_not_normalized - R_truth_not_normalized.min()) / (R_truth_not_normalized.max() - R_truth_not_normalized.min())

    return R_truth, T_truth, V_truth

def music_market(R_truth, user_properties, seed=None):    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize an empty matrix (all ratings zero initially)
    R_observed = np.zeros((I, J))
    # Create a mask to track which entries have been observed (all False initially)
    R_mask = np.zeros((I, J), dtype=bool)

    play_count = np.zeros((I, J), dtype=int)

    # Make sure there are available songs
    for i in range(I):
        j = np.random.choice(J)
        true_score = R_truth[i, j]
        R_observed[i, j] = true_score # Randomly select from available songs
        R_mask[i, j] = True
        play_count[i,j] += 1  # Mark this entry as observed

    current_density = np.mean(R_mask)
    print(f"Step τ=1: Initial density = {current_density:.4f}")

    # For storing the sequence of recommendations
    recommendations_history = []
    
    # Initialize factor matrices
    
    T_hat = np.random.rand(I, H)
    V_hat = np.random.rand(J, H)

    # Iterative recommendation process for τ = 2, 3, ... until target density reached
    tau = 2
    while current_density < target_density:
        print(f"\n--- Starting Step τ={tau} ---")
        
        # Get indices of observed elements for SGD
        nonzero_i, nonzero_j = np.where(R_mask)
        observed_values = R_observed[nonzero_i, nonzero_j]
        n_observed = len(nonzero_i)

        for _ in range(iterations):
            # Shuffle indices instead of copying data
            perm = np.random.permutation(n_observed)
            
            for idx in perm:
                i, j = nonzero_i[idx], nonzero_j[idx]
                r_ij = observed_values[idx]
                r_ij_pred = np.dot(T_hat[i], V_hat[j])
                e_ij = r_ij - r_ij_pred
                
                t_i_old = T_hat[i].copy()
                T_hat[i] += learning_rate * (e_ij * V_hat[j] - lambda_t * T_hat[i])
                V_hat[j] += learning_rate * (e_ij * t_i_old - lambda_v * V_hat[j])
        
        R_predicted = T_hat @ V_hat.T
        
        # For each consumer, find the product with the highest predicted rating
        # among products they haven't tried yet and that are available
        recommendations = []
        
        for i in range(I):
            
            exploration_prob = user_properties['explorer_score'][i]

            if np.random.rand() < exploration_prob:  # Explore a random unobserved song
                # Get unobserved songs that are also available
                unobserved_songs = np.where(R_mask[i] == False)[0]
                
                if len(unobserved_songs) > 0:
                    j_star = np.random.choice(unobserved_songs)
                else:
                    # No available unobserved songs, proceed with exploitation
                    user_predictions = R_predicted[i].copy()
                    
                    j_star = np.argmax(user_predictions)
            else:
                # Create a copy of predictions
                user_predictions = R_predicted[i].copy()
                
                j_star = np.argmax(user_predictions)
                
            recommendations.append((i, j_star))
            # User tries the recommended product and rates it
            true_score = R_truth[i, j_star]
            R_observed[i, j_star] = true_score
            R_mask[i, j_star] = True  # Mark as observed
            play_count[i,j_star] += 1  # Increment play count

        recommendations_history.append(recommendations)
        
        # Calculate and print density
        current_density = np.mean(R_mask)
        print(f"Step τ={tau}: Density = {current_density:.4f}")
        
        tau += 1
    
    print(f"\nFinished after {tau-1} steps with density {current_density:.4f}")
    
    return T_hat, V_hat, R_observed, R_mask, recommendations_history, play_count

'''''''''''''''''''''album creation'''''''''''''''''''''''''''

J_new = J + album_len * num_of_albums

distinct_hit_prob = 0.3 

def create_albums(J_new, album_len, num_of_albums):
    albums = {}
    # Reserve space for albums
    album_start = J_new - (num_of_albums * album_len)
    
    for a in range(num_of_albums):
        start_idx = album_start + (a * album_len)
        albums[a] = list(range(start_idx, start_idx + album_len))
    
    return albums

def expand_matrices(R_truth, T_hat, V_hat, R_observed, R_mask, new_song_count):

    # Get current dimensions
    I, _ = T_hat.shape
    J, _ = V_hat.shape
    
    R_truth_new_songs, T_truth_new, V_truth_new_songs = initialize_truth_matrix(I, new_song_count, H)
    
    # Expand V_hat using the properly initialized V_truth for new songs, with cold start penalty
    V_hat_expanded = np.zeros((J + new_song_count, H))
    V_hat_expanded[:J, :] = V_hat
    V_hat_expanded[J:, :] = np.random.rand(new_song_count, H)
    
    # Expand R_truth using the properly generated ratings for new songs
    R_truth_expanded = np.zeros((I, J + new_song_count))
    R_truth_expanded[:, :J] = R_truth
    R_truth_expanded[:, J:] = R_truth_new_songs
    
    # R_observed and R_mask expansion (keeping your existing logic)
    R_observed_expanded = np.zeros((I, J + new_song_count))
    R_observed_expanded[:, :J] = R_observed
    
    R_mask_expanded = np.zeros((I, J + new_song_count), dtype=bool)
    R_mask_expanded[:, :J] = R_mask
    
    return R_truth_expanded, T_hat, V_hat_expanded, R_observed_expanded, R_mask_expanded

albums = create_albums(J_new, album_len, num_of_albums)

def create_album_fanbases(albums, I, seed=None):

    # Set the seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    album_fanbases = {}
    for album_id in albums:
        #fanbase_size = int(I * np.random.uniform(0.005, 0.02))
        
        #'''
        if album_id == 1:
            fanbase_size = 0
        else:
            fanbase_size = int(I * np.random.uniform(0.05, 0.10))
        #'''   
        
        # Randomly select users for this album's fanbase
        fanbase_users = np.random.choice(I, fanbase_size, replace=False)
        album_fanbases[album_id] = fanbase_users
    
    return album_fanbases

album_fanbases = create_album_fanbases(albums, I, seed=None)

'''''''''''''''''''''starting postions'''''''''''''''''''''''''''

def starting_positions(album_fanbases, albums, J_new, J_original, I, T_hat, V_hat, R_observed, R_mask, R_truth, user_properties, 
                          fan_properties, recommendation_adherence, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    # Track play counts for each user-song combination
    play_count = np.zeros((I, J_new), dtype=int)
    
    # Lists to store T_hat and V_hat for each time period
    T_hat_history = []
    V_hat_history = []
    
    # Song availability tracker (all old songs available initially)
    song_available = np.ones(J_original, dtype=bool)
    song_available_expanded = np.concatenate([song_available, np.zeros(J_new - J_original, dtype=bool)])
    
    # Recommendations history
    recommendations_history = []
    
    # Before releasing songs, check if matrices need expansion
    current_J = R_observed.shape[1]
    if current_J < J_new:
        # Expand matrices as needed
        R_truth, T_hat, V_hat, R_observed, R_mask = expand_matrices(
            R_truth, T_hat, V_hat, R_observed, R_mask, J_new - current_J
        )
        # Also expand play_count matrix
        play_count_expanded = np.zeros((I, J_new), dtype=int)
        play_count_expanded[:, :play_count.shape[1]] = play_count
        play_count = play_count_expanded

    print(f"\n--- Starting Positions Captured ---")

    album_ids = list(albums.keys())
    np.random.shuffle(album_ids)
        
    # Store T_hat and V_hat after this time period
    T_hat_history.append(T_hat.copy())
    V_hat_history.append(V_hat.copy())

    # Calculate and print analytics
    item_popularity_new = np.sum(R_mask, axis=0)
    
    # For new songs only
    new_song_indices = []
    for album_id in albums:
        new_song_indices.extend(albums[album_id])
        
    print("New songs unique listeners:", item_popularity_new[new_song_indices])
    print("New songs total plays:", np.sum(play_count, axis=0)[new_song_indices])
    
    current_density = np.mean(R_mask)
    print(f"Density = {current_density:.4f}")
    
    return R_truth, T_hat, V_hat, R_observed, R_mask, recommendations_history, play_count, album_fanbases, T_hat_history, V_hat_history

'''''''''''''''''''''gradual'''''''''''''''''''''''''''

time_periods_gradual = 8 # Number of time periods for gradual release

def album_release_gradual(album_fanbases, albums, J_new, J_original, I, T_hat, V_hat, R_observed, R_mask, R_truth, user_properties, 
                          fan_properties, recommendation_adherence, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    # Track play counts for each user-song combination
    play_count = np.zeros((I, J_new), dtype=int)
    
    # Lists to store T_hat and V_hat for each time period
    T_hat_history = []
    V_hat_history = []
    
    # Song availability tracker (all old songs available initially)
    song_available = np.ones(J_original, dtype=bool)
    song_available_expanded = np.concatenate([song_available, np.zeros(J_new - J_original, dtype=bool)])
    
    # Recommendations history
    recommendations_history = []
    
    # Before releasing songs, check if matrices need expansion
    current_J = R_observed.shape[1]
    if current_J < J_new:
        # Expand matrices as needed
        R_truth, T_hat, V_hat, R_observed, R_mask = expand_matrices(
            R_truth, T_hat, V_hat, R_observed, R_mask, J_new - current_J
        )
        # Also expand play_count matrix
        play_count_expanded = np.zeros((I, J_new), dtype=int)
        play_count_expanded[:, :play_count.shape[1]] = play_count
        play_count = play_count_expanded

    # Time periods simulation
    for t in range(1, time_periods_gradual + 1):
        print(f"\n--- Time Period {t} ---")

        users_who_rated_new_songs_this_period = set()

        recommendations = []

        # Album release logic - same as original
        if t == 1:  # First singles from albums
            album_ids = list(albums.keys())
            np.random.shuffle(album_ids)
            for album_id in album_ids:
                song_index = 0
                song_id = albums[album_id][song_index]
                song_available_expanded[song_id] = True
                
                # Fanbase listens and rates
                for user in album_fanbases[album_id]:
                    true_score = R_truth[user, song_id]

                    R_observed[user, song_id] = np.clip(true_score + fan_properties['fan_bias'][user], 0, 1)

                    #Bias towards artist
                    R_truth[user, song_id] = np.clip(true_score + fan_properties['fan_bias'][user], 0, 1)

                    R_mask[user, song_id] = True
                    play_count[user, song_id] += 1
                    recommendations.append((user, song_id, song_id, True))
                
                print(f"Album {album_id}, Song {song_index} released")
        
        elif t == 4:  # Remaining songs from albums
            album_ids = list(albums.keys())
            np.random.shuffle(album_ids)
            for album_id in album_ids:
                for song_index in range(1, album_len):
                    song_id = albums[album_id][song_index]
                    song_available_expanded[song_id] = True
                    
                    # Fanbase listens and rates
                    for user in album_fanbases[album_id]:
                        true_score = R_truth[user, song_id]
                        R_observed[user, song_id] = np.clip(true_score + fan_properties['fan_bias'][user], 0, 1)

                        #Sunk cost fallacy towards artist
                        R_truth[user, song_id] = np.clip(true_score + fan_properties['fan_bias'][user], 0, 1)

                        R_mask[user, song_id] = True
                        play_count[user, song_id] += 1
                        recommendations.append((user, song_id, song_id, True))  # Record recommendation

                print(f"Album {album_id}, Songs 1-2 released")

        if t == 1 or t == 4:  # When new songs are released
            for album_id in albums.keys():
                for user in album_fanbases[album_id]:
                    users_who_rated_new_songs_this_period.add(user)

        # Run recommendation algorithm iterations after each time period
        # Get indices of observed elements for SGD

        nonzero_i, nonzero_j = np.where(R_mask)
        observed_values = R_observed[nonzero_i, nonzero_j]
        n_observed = len(nonzero_i)

        for _ in range(iterations):
            # Shuffle indices instead of copying data
            perm = np.random.permutation(n_observed)
            
            for idx in perm:
                i, j = nonzero_i[idx], nonzero_j[idx]
                r_ij = observed_values[idx]
                r_ij_pred = np.dot(T_hat[i], V_hat[j])
                e_ij = r_ij - r_ij_pred
                
                t_i_old = T_hat[i].copy()
                T_hat[i] += learning_rate * (e_ij * V_hat[j] - lambda_t * T_hat[i])
                V_hat[j] += learning_rate * (e_ij * t_i_old - lambda_v * V_hat[j])
            
        # Calculate the full predicted matrix
        R_predicted = T_hat @ V_hat.T
        R_predicted_mean = np.mean(R_predicted)
        
        # Recommendation process

        for i in range(I):

            if i in users_who_rated_new_songs_this_period:
                continue

            # Apply the reduction to the original exploration probability
            exploration_prob = (user_properties['explorer_score'][i])
            
            # Check if user has rated a recently released album song above the mean
            should_recommend_upcoming_album = False
            existing_album_songs = []
            
            # Determine which albums will be released in the upcoming time period
            upcoming_album_ids = []
            this_period = t
            
            if this_period == 4:  # Rest of first albums
                upcoming_album_ids = list(albums.keys())
                np.random.shuffle(upcoming_album_ids)
            
            qualifying_albums = []

            # Check recent ratings for released songs from upcoming albums
            for album_id in upcoming_album_ids:
                album_songs = albums[album_id] # Get songs from this album
                for song_id in album_songs:
                    # Check if this song is released and has been rated by the user
                    if song_available_expanded[song_id] and R_mask[i, song_id]:
                        # Check if rating is above mean
                        if R_observed[i, song_id] > R_predicted_mean:
                            # Find unreleased songs from this album for recommendation
                            qualifying_albums.append(album_id)
                            break

            if qualifying_albums:
                chosen_album = np.random.choice(qualifying_albums)
                unreleased_songs = [s for s in albums[chosen_album] if song_available_expanded[s]]
                if unreleased_songs:
                    should_recommend_upcoming_album = True
                    existing_album_songs.extend(unreleased_songs)

            # Generate the recommendation (which may or may not be followed)
            if should_recommend_upcoming_album and existing_album_songs and np.random.rand() > distinct_hit_prob:
                # User likes a song from an album with upcoming releases
                # Recommend an unreleased song from that album (preparation for future release)
                j_star_recommendation = np.random.choice(existing_album_songs)
            else:
                user_predictions = R_predicted[i].copy()
                if np.random.rand() < exploration_prob:
                    for j in range(J_new):
                        if not song_available_expanded[j]:
                            user_predictions[j] = float('-inf')  # Explore
                    j_star_recommendation = np.random.choice(np.argsort(user_predictions)[int(-(J_new/20)):])
                else:  # Exploit
                    # Mask out unavailable items
                    for j in range(J_new):
                        if not song_available_expanded[j]:
                            user_predictions[j] = float('-inf')
                    
                    # Add bias to songs the user has already rated highly
                    rating_threshold = R_predicted_mean  # Songs rated above this threshold get boosted
                    
                    for j in range(J_new):
                        if R_mask[i, j] and song_available_expanded[j]:
                            # If the user has rated this song above threshold
                            if R_observed[i, j] > rating_threshold:
                                # Add a boost proportional to how much they liked it
                                boost = (R_observed[i, j] - rating_threshold) * rating_boost
                                user_predictions[j] += boost
                    
                    j_star_recommendation = np.argmax(user_predictions)
            
            follows_recommendation = np.random.rand() < recommendation_adherence[i]
            
            if follows_recommendation:
                # User follows the recommendation
                j_star = j_star_recommendation
            else:
                # Find the songs the user has already rated
                rated_songs = np.where(R_mask[i] & song_available_expanded)[0]
                
                if len(rated_songs) > 0:
                    # Get ratings for songs the user has rated
                    user_ratings = R_observed[i, rated_songs]
                    
                    # Make sure all ratings are positive by shifting to minimum 0.01. This ensures we don't have negative or zero probabilities.
                    adjusted_ratings = np.maximum(user_ratings, 0.01)
                    
                    # Calculate probability weights based on ratings
                    # Higher ratings have higher probability of being picked
                    selection_weights = adjusted_ratings / np.sum(adjusted_ratings)
                    
                    # Choose one of their previously rated songs based on their ratings
                    j_star = np.random.choice(rated_songs, p=selection_weights)
                else:
                    # No rated songs yet, pick a random available song
                    j_star = j_star_recommendation

            
            # Record recommendation and what was actually listened to
            recommendations.append((i, j_star_recommendation, j_star, follows_recommendation))
            
            # User listens to the selected song
            play_count[i, j_star] += 1
            
            true_score = R_truth[i, j_star]
            R_observed[i, j_star] = np.clip(true_score, 0, 1)
            
            if not R_mask[i, j_star]:
                R_mask[i, j_star] = True
        
        recommendations_history.append(recommendations)

        # Store T_hat and V_hat after this time period
        T_hat_history.append(T_hat.copy())
        V_hat_history.append(V_hat.copy())

        # Calculate and print analytics
        item_popularity_new = np.sum(R_mask, axis=0)
        
        # For new songs only
        new_song_indices = []
        for album_id in albums:
            new_song_indices.extend(albums[album_id])
            
        print("New songs unique listeners:", item_popularity_new[new_song_indices])
        print("New songs total plays:", np.sum(play_count, axis=0)[new_song_indices])
        
        # NEW: Calculate recommendation adherence rate
        recommendations_followed = sum(1 for _, _, _, followed in recommendations if followed)
        adherence_rate = recommendations_followed / len(recommendations)
        print(f"Recommendation adherence rate: {adherence_rate:.2f}")
        
        # For each new song, calculate repeat listen rate
        print("\nRepeat listen stats for new songs:")
        for album_id in albums:
            for idx, song_id in enumerate(albums[album_id]):
                if song_available_expanded[song_id]:
                    unique_listeners = item_popularity_new[song_id]
                    total_plays = np.sum(play_count[:, song_id])
                    if unique_listeners > 0:
                        avg_plays = total_plays / unique_listeners
                        print(f"Album {album_id}, Song {idx}: {unique_listeners} listeners, {total_plays} plays, {avg_plays:.2f} plays/listener")
        
        current_density = np.mean(R_mask)
        print(f"Density = {current_density:.4f}")
    
    return R_truth, T_hat, V_hat, R_observed, R_mask, recommendations_history, play_count, album_fanbases, T_hat_history, V_hat_history

'''''''''''''''''''''immediate'''''''''''''''''''''''''''

time_periods_immediate = 6

def album_release_immediate(album_fanbases, albums, J_new, J_original, I, T_hat, V_hat, R_observed, R_mask, R_truth, user_properties, 
                          fan_properties, recommendation_adherence, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    # Track play counts for each user-song combination
    play_count = np.zeros((I, J_new), dtype=int)
    
    # Lists to store T_hat and V_hat for each time period
    T_hat_history = []
    V_hat_history = []
    
    # Song availability tracker (all old songs available initially)
    song_available = np.ones(J_original, dtype=bool)
    song_available_expanded = np.concatenate([song_available, np.zeros(J_new - J_original, dtype=bool)])
    
    # Recommendations history
    recommendations_history = []
    
    # Before releasing songs, check if matrices need expansion
    current_J = R_observed.shape[1]
    if current_J < J_new:
        # Expand matrices as needed
        R_truth, T_hat, V_hat, R_observed, R_mask = expand_matrices(
            R_truth, T_hat, V_hat, R_observed, R_mask, J_new - current_J
        )
        # Also expand play_count matrix
        play_count_expanded = np.zeros((I, J_new), dtype=int)
        play_count_expanded[:, :play_count.shape[1]] = play_count
        play_count = play_count_expanded


    # Time periods simulation
    for t in range(1, time_periods_immediate + 1):
        print(f"\n--- Time Period {t} ---")

        users_who_rated_new_songs_this_period = set()

        recommendations = []

        # Album release logic - same as original
        
        if t == 1:  # All songs from first half of albums
            album_ids = list(albums.keys())
            np.random.shuffle(album_ids)
            for album_id in album_ids:
                for song_index in range(0, album_len):
                    song_id = albums[album_id][song_index]
                    song_available_expanded[song_id] = True
                    
                    # Fanbase listens and rates
                    for user in album_fanbases[album_id]:
                        true_score = R_truth[user, song_id]
                        R_observed[user, song_id] = np.clip(true_score + fan_properties['fan_bias'][user], 0, 1)

                        #Sunk cost fallacy towards artist
                        R_truth[user, song_id] = np.clip(true_score + fan_properties['fan_bias'][user], 0, 1)

                        R_mask[user, song_id] = True
                        play_count[user, song_id] += 1
                        recommendations.append((user, song_id, song_id, True))

                print(f"Album {album_id} released")

        if t == 1:  # When new songs are released
            for album_id in albums.keys():
                for user in album_fanbases[album_id]:
                    users_who_rated_new_songs_this_period.add(user)

        # Run recommendation algorithm iterations after each time period
        # Get indices of observed elements for SGD

        nonzero_i, nonzero_j = np.where(R_mask)
        observed_values = R_observed[nonzero_i, nonzero_j]
        n_observed = len(nonzero_i)

        for _ in range(iterations):
            # Shuffle indices instead of copying data
            perm = np.random.permutation(n_observed)
            
            for idx in perm:
                i, j = nonzero_i[idx], nonzero_j[idx]
                r_ij = observed_values[idx]
                r_ij_pred = np.dot(T_hat[i], V_hat[j])
                e_ij = r_ij - r_ij_pred
                
                t_i_old = T_hat[i].copy()
                T_hat[i] += learning_rate * (e_ij * V_hat[j] - lambda_t * T_hat[i])
                V_hat[j] += learning_rate * (e_ij * t_i_old - lambda_v * V_hat[j])
            
        # Calculate the full predicted matrix
        R_predicted = T_hat @ V_hat.T
        R_predicted_mean = np.mean(R_predicted)

        # Recommendation process

        for i in range(I):

            if i in users_who_rated_new_songs_this_period:
                continue

            # Apply the reduction to the original exploration probability
            exploration_prob = (user_properties['explorer_score'][i])
            
            # Check if user has rated a recently released album song above the mean
            should_recommend_album = False
            existing_album_songs = []
            
            # Determine which albums will be released in the upcoming time period
            existing_album_ids = list(albums.keys())
            np.random.shuffle(existing_album_ids)
            
            qualifying_albums = []

            # Check recent ratings for released songs from upcoming albums
            for album_id in existing_album_ids:
                album_songs = albums[album_id] # Get songs from this album
                for song_id in album_songs:
                    # Check if this song is released and has been rated by the user
                    if song_available_expanded[song_id] and R_mask[i, song_id]:
                        # Check if rating is above mean
                        if R_observed[i, song_id] > R_predicted_mean:
                            # Find released songs from this album for recommendation
                            qualifying_albums.append(album_id)
                            break

            if qualifying_albums:
                chosen_album = np.random.choice(qualifying_albums)
                more_album_songs = [s for s in albums[chosen_album] if song_available_expanded[s]]
                if more_album_songs:
                    should_recommend_album = True
                    existing_album_songs.extend(more_album_songs)
            
            # Generate the recommendation (which may or may not be followed)
            if should_recommend_album and existing_album_songs and np.random.rand() > distinct_hit_prob:
                # User likes a song from an album with upcoming releases
                # Recommend an unreleased song from that album (preparation for future release)
                j_star_recommendation = np.random.choice(existing_album_songs)
            else:
                user_predictions = R_predicted[i].copy()
                if np.random.rand() < exploration_prob:
                    for j in range(J_new):
                        if not song_available_expanded[j]:
                            user_predictions[j] = float('-inf')  # Explore
                    j_star_recommendation = np.random.choice(np.argsort(user_predictions)[int(-(J_new/20)):])
                else:  # Exploit
                    # Mask out unavailable items
                    for j in range(J_new):
                        if not song_available_expanded[j]:
                            user_predictions[j] = float('-inf')
                    
                    # Add bias to songs the user has already rated highly
                    rating_threshold = R_predicted_mean  # Songs rated above this threshold get boosted
                    
                    for j in range(J_new):
                        if R_mask[i, j] and song_available_expanded[j]:
                            # If the user has rated this song above threshold
                            if R_observed[i, j] > rating_threshold:
                                # Add a boost proportional to how much they liked it
                                boost = (R_observed[i, j] - rating_threshold) * rating_boost
                                user_predictions[j] += boost
                    
                    j_star_recommendation = np.argmax(user_predictions)
            
            follows_recommendation = np.random.rand() < recommendation_adherence[i]
            
            if follows_recommendation:
                # User follows the recommendation
                j_star = j_star_recommendation
            else:
                # Find the songs the user has already rated
                rated_songs = np.where(R_mask[i] & song_available_expanded)[0]
                
                if len(rated_songs) > 0:
                    # Get ratings for songs the user has rated
                    user_ratings = R_observed[i, rated_songs]
                    
                    # Make sure all ratings are positive by shifting to minimum 0.01. This ensures we don't have negative or zero probabilities.
                    adjusted_ratings = np.maximum(user_ratings, 0.01)
                    
                    # Calculate probability weights based on ratings
                    # Higher ratings have higher probability of being picked
                    selection_weights = adjusted_ratings / np.sum(adjusted_ratings)
                    
                    # Choose one of their previously rated songs based on their ratings
                    j_star = np.random.choice(rated_songs, p=selection_weights)
                else:
                    # No rated songs yet, pick a random available song
                    j_star = j_star_recommendation

            
            # Record recommendation and what was actually listened to
            recommendations.append((i, j_star_recommendation, j_star, follows_recommendation))
            
            # User listens to the selected song
            play_count[i, j_star] += 1
            
            true_score = R_truth[i, j_star]
            R_observed[i, j_star] = np.clip(true_score, 0, 1)
            
            if not R_mask[i, j_star]:
                R_mask[i, j_star] = True
        
        recommendations_history.append(recommendations)

        # Store T_hat and V_hat after this time period
        T_hat_history.append(T_hat.copy())
        V_hat_history.append(V_hat.copy())

        # Calculate and print analytics
        item_popularity_new = np.sum(R_mask, axis=0)
        
        # For new songs only
        new_song_indices = []
        for album_id in albums:
            new_song_indices.extend(albums[album_id])
            
        print("New songs unique listeners:", item_popularity_new[new_song_indices])
        print("New songs total plays:", np.sum(play_count, axis=0)[new_song_indices])
        
        # NEW: Calculate recommendation adherence rate
        recommendations_followed = sum(1 for _, _, _, followed in recommendations if followed)
        adherence_rate = recommendations_followed / len(recommendations)
        print(f"Recommendation adherence rate: {adherence_rate:.2f}")
        
        # For each new song, calculate repeat listen rate
        print("\nRepeat listen stats for new songs:")
        for album_id in albums:
            for idx, song_id in enumerate(albums[album_id]):
                if song_available_expanded[song_id]:
                    unique_listeners = item_popularity_new[song_id]
                    total_plays = np.sum(play_count[:, song_id])
                    if unique_listeners > 0:
                        avg_plays = total_plays / unique_listeners
                        print(f"Album {album_id}, Song {idx}: {unique_listeners} listeners, {total_plays} plays, {avg_plays:.2f} plays/listener")
        
        current_density = np.mean(R_mask)
        print(f"Density = {current_density:.4f}")
    
    return R_truth, T_hat, V_hat, R_observed, R_mask, recommendations_history, play_count, album_fanbases, T_hat_history, V_hat_history

'''''''''''''''''''''if main'''''''''''''''''''''''''''

if __name__ == "__main__":

    #Exploration probability
    user_properties = {
        'explorer_score': np.random.beta(1, 22, size=I)       # How likely they are to explore new content
        }

    #Sunk cost fallacy
    fan_properties = {
        'fan_bias': np.random.beta(1, 22, size=I)       
        }

    #Recommendation adherence property - how likely a user is to follow recommendations. Lower values mean users more frequently choose their own favorites instead
    recommendation_adherence = np.random.beta(2, 8, size=I)  

    R_truth, T_truth, V_truth = initialize_truth_matrix(I, J, H)

    T_hat_1, V_hat_1, R_observed_1, R_mask_1, recommendations_history_1, play_count_1 = music_market(R_truth, seed=None)

    R_truth_gradual, T_hat_gradual, V_hat_gradual, R_observed_gradual, R_mask_gradual, recommendations_history_gradual, play_count_gradual, album_fanbases_gradual = album_release_gradual( 
    album_fanbases, 
    albums, 
    J_new,
    J,
    I, 
    T_hat_1,
    V_hat_1, 
    R_observed_1, 
    R_mask_1,
    R_truth,  
    seed=None
    )

    print(np.sum((T_hat_gradual @ V_hat_gradual.T - R_truth_gradual) ** 2))
    print(np.mean(R_observed_gradual[R_observed_gradual != 0]))

    print(np.sum(play_count_gradual, axis=0))

    for album_id in album_fanbases:
        print(f"Album {album_id} has {len(album_fanbases[album_id])} fans")

    R_truth_immediate, T_hat_immediate, V_hat_immediate, R_observed_immediate, R_mask_immediate, recommendations_history_immediate, play_count_immediate, album_fanbases_immediate = album_release_immediate( 
    album_fanbases, 
    albums, 
    J_new,
    J,
    I, 
    T_hat_1,
    V_hat_1, 
    R_observed_1, 
    R_mask_1,
    R_truth,  
    seed=None
    )

    print(np.sum((T_hat_immediate @ V_hat_immediate.T - R_truth_immediate) ** 2))
    print(np.mean(R_observed_immediate[R_observed_immediate != 0]))

    print(np.sum(play_count_immediate, axis=0))
