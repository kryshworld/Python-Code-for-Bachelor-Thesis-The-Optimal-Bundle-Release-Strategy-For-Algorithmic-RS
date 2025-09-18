import numpy as np
import itertools
print("Numpy imported successfully")

# Parameters
I = 24030 # Number of users
J = 801  # Number of items
H = 2 # Latent factors
iterations = 50  # SGD iterations for each step
target_density = 0.012  # Continue until this density is reached
max_tau_steps = 100   # Safety limit to prevent infinite loops

# Parameter grids to test
lambda_t_values = [0.01, 0.05, 0.1, 0.15, 0.2]
lambda_v_values = [0.01, 0.05, 0.1, 0.15, 0.2]
learning_rate_values = [0.01, 0.05, 0.1, 0.15, 0.2]

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

def music_market(R_truth, user_properties, lambda_t, lambda_v, learning_rate, seed=None, verbose=False):    
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
    if verbose:
        print(f"Step τ=1: Initial density = {current_density:.4f}")

    # For storing the sequence of recommendations
    recommendations_history = []
    
    # Initialize factor matrices
    T_hat = np.random.rand(I, H)
    V_hat = np.random.rand(J, H)

    # Iterative recommendation process for τ = 2, 3, ... until target density reached
    tau = 2
    while current_density < target_density and tau <= max_tau_steps:
        if verbose:
            print(f"\n--- Starting Step τ={tau} ---")
        
        # Get indices of observed elements for SGD (using R_mask to identify them)
        nonzero_i, nonzero_j = np.where(R_mask)
        nonzero_indices = list(zip(nonzero_i, nonzero_j))
        
        # Train the model using current observations
        for it in range(iterations):
            # Shuffle indices for stochastic updates
            np.random.shuffle(nonzero_indices)
            
            # Loop through each observed element
            for i, j in nonzero_indices:
                # Only use entries that are marked as observed in R_mask
                if R_mask[i, j]:
                    # Compute error for this specific rating
                    r_ij = R_observed[i, j]
                    r_ij_pred = np.dot(T_hat[i], V_hat[j])
                    e_ij = r_ij - r_ij_pred
                    
                    # Update factors for this rating
                    t_i_old = T_hat[i].copy()
                    T_hat[i] += learning_rate * (e_ij * V_hat[j] - lambda_t * T_hat[i])
                    V_hat[j] += learning_rate * (e_ij * t_i_old - lambda_v * V_hat[j])
        
        # Calculate the full predicted matrix
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
        if verbose:
            print(f"Step τ={tau}: Density = {current_density:.4f}")
        
        tau += 1
    
    # Calculate final MSE
    R_predicted_final = T_hat @ V_hat.T
    final_mse = np.sum((R_observed[R_mask] - R_predicted_final[R_mask])**2)
    
    if verbose:
        print(f"\nFinished after {tau-1} steps with density {current_density:.4f}")
        print(f"Final MSE: {final_mse:.6f}")
    
    return final_mse, tau-1, current_density

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

# Set seed for reproducibility
run_seed = 909109
print(f"Using seed: {run_seed}")

# Initialize truth matrix once
R_truth, _, _ = initialize_truth_matrix(I, J, H)
user_properties, fan_properties, recommendation_adherence = generate_user_properties(I, seed=run_seed)

# Grid search
results = []
total_combinations = len(lambda_t_values) * len(lambda_v_values) * len(learning_rate_values)
combination_count = 0

print(f"\nStarting grid search with {total_combinations} combinations...")
print("Format: lambda_t, lambda_v, learning_rate -> final_mse, steps, final_density")
print("-" * 80)

for lambda_t, lambda_v, learning_rate in itertools.product(lambda_t_values, lambda_v_values, learning_rate_values):
    combination_count += 1
    
    try:
        final_mse, steps, final_density = music_market(
            R_truth, user_properties, lambda_t, lambda_v, learning_rate, 
            seed=run_seed, verbose=False
        )
        
        results.append({
            'lambda_t': lambda_t,
            'lambda_v': lambda_v, 
            'learning_rate': learning_rate,
            'final_mse': final_mse,
            'steps': steps,
            'final_density': final_density
        })
        
        print(f"[{combination_count:2d}/{total_combinations}] λt={lambda_t:.2f}, λv={lambda_v:.2f}, lr={learning_rate:.2f} -> MSE={final_mse:.6f}, steps={steps}, density={final_density:.4f}")
        
    except Exception as e:
        print(f"[{combination_count:2d}/{total_combinations}] λt={lambda_t:.2f}, λv={lambda_v:.2f}, lr={learning_rate:.2f} -> ERROR: {str(e)}")

# Find best result
if results:
    best_result = min(results, key=lambda x: x['final_mse'])
    
    print("\n" + "="*80)
    print("BEST RESULT:")
    print(f"lambda_t: {best_result['lambda_t']}")
    print(f"lambda_v: {best_result['lambda_v']}")
    print(f"learning_rate: {best_result['learning_rate']}")
    print(f"Final MSE: {best_result['final_mse']:.6f}")
    print(f"Steps: {best_result['steps']}")
    print(f"Final density: {best_result['final_density']:.4f}")
    
    print("\nTop 5 results by MSE:")
    sorted_results = sorted(results, key=lambda x: x['final_mse'])
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. λt={result['lambda_t']:.2f}, λv={result['lambda_v']:.2f}, lr={result['learning_rate']:.2f} -> MSE={result['final_mse']:.6f}")