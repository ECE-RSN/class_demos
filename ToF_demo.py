import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import random

class TOFLocalizer:
    def __init__(self, speed_of_sound=343.0):
        """Initialize with speed of sound in m/s"""
        self.speed_of_sound = speed_of_sound
    
    def distance_between_points(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def time_of_flight(self, transmitter_pos, receiver_pos, noise_std=0.0):
        """
        Calculate time of flight from transmitter to receiver
        Add Gaussian noise if noise_std > 0 (in milliseconds)
        """
        distance = self.distance_between_points(transmitter_pos, receiver_pos)
        tof = distance / self.speed_of_sound
        
        # Add noise if specified
        if noise_std > 0:
            noise = np.random.normal(0, noise_std / 1000.0)  # Convert ms to seconds
            tof = max(0, tof + noise)  # Ensure non-negative
        
        return tof
    
    def localize(self, tx1_pos, tx2_pos, tof1, tof2):
        """
        Localize receiver using two transmitters and their time-of-flight measurements
        """
        # Convert time to distance
        d1 = tof1 * self.speed_of_sound
        d2 = tof2 * self.speed_of_sound
        
        # System of circle equations:
        # (x - tx1_x)² + (y - tx1_y)² = d1²
        # (x - tx2_x)² + (y - tx2_y)² = d2²
        def equations(pos):
            x, y = pos
            eq1 = (x - tx1_pos[0])**2 + (y - tx1_pos[1])**2 - d1**2
            eq2 = (x - tx2_pos[0])**2 + (y - tx2_pos[1])**2 - d2**2
            return [eq1, eq2]
        
        # Initial guess at midpoint
        initial_guess = [(tx1_pos[0] + tx2_pos[0])/2, (tx1_pos[1] + tx2_pos[1])/2]
        
        # Solve the system
        solution = fsolve(equations, initial_guess)
        return solution
    
    def visualize(self, tx1_pos, tx2_pos, actual_receiver, estimated_receiver, tof1, tof2):
        """Create visualization of the localization scenario"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot transmitters
        ax.plot(tx1_pos[0], tx1_pos[1], 'rs', markersize=15, label='Transmitter 1', markeredgecolor='darkred')
        ax.plot(tx2_pos[0], tx2_pos[1], 'bs', markersize=15, label='Transmitter 2', markeredgecolor='darkblue')
        
        # Plot receivers
        ax.plot(actual_receiver[0], actual_receiver[1], 'go', markersize=12, 
               label='True Position', markeredgecolor='darkgreen')
        ax.plot(estimated_receiver[0], estimated_receiver[1], 'g^', markersize=12, 
               label='Estimated Position', markerfacecolor='lightgreen', markeredgecolor='darkgreen')
        
        # Draw distance circles
        d1 = tof1 * self.speed_of_sound
        d2 = tof2 * self.speed_of_sound
        
        circle1 = plt.Circle(tx1_pos, d1, fill=False, color='red', linestyle='--', linewidth=2,
                           label=f'Distance Circle 1 ({d1:.1f}m)')
        circle2 = plt.Circle(tx2_pos, d2, fill=False, color='blue', linestyle='--', linewidth=2,
                           label=f'Distance Circle 2 ({d2:.1f}m)')
        
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        # Draw signal paths
        ax.plot([tx1_pos[0], actual_receiver[0]], [tx1_pos[1], actual_receiver[1]], 
               'r:', alpha=0.6, linewidth=2)
        ax.plot([tx2_pos[0], actual_receiver[0]], [tx2_pos[1], actual_receiver[1]], 
               'b:', alpha=0.6, linewidth=2)
        
        # Add timing annotations
        mid1 = [(tx1_pos[0] + actual_receiver[0])/2, (tx1_pos[1] + actual_receiver[1])/2]
        mid2 = [(tx2_pos[0] + actual_receiver[0])/2, (tx2_pos[1] + actual_receiver[1])/2]
        
        ax.annotate(f'{tof1*1000:.2f} ms', mid1, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
        ax.annotate(f'{tof2*1000:.2f} ms', mid2, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='blue'))
        
        # Calculate and display error
        error = self.distance_between_points(actual_receiver, estimated_receiver)
        ax.text(0.02, 0.98, f'Localization Error: {error:.2f} m', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_xlim(-10, 110)
        ax.set_ylim(-10, 110)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Time-of-Flight Localization')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()

def run_simulation():
    """Main simulation function"""
    print("=" * 50)
    print("TIME-OF-FLIGHT LOCALIZATION SIMULATOR")
    print("=" * 50)
    
    # Initialize system
    localizer = TOFLocalizer()
    
    # Set up transmitter positions
    tx1_pos = (20, 20)  # Bottom-left transmitter
    tx2_pos = (80, 80)  # Top-right transmitter
    
    print(f"Transmitter 1 at: {tx1_pos}")
    print(f"Transmitter 2 at: {tx2_pos}")
    print(f"Speed of sound: {localizer.speed_of_sound} m/s")
    print()
    
    # Get receiver position from user
    while True:
        try:
            rx_input = input("Enter receiver position (x,y) [e.g., 50,60]: ")
            rx_x, rx_y = map(float, rx_input.split(','))
            actual_receiver = (rx_x, rx_y)
            break
        except ValueError:
            print("Invalid format. Please use 'x,y' with numbers.")
    
    # Get noise level from user
    while True:
        try:
            noise_input = input("Enter noise level in milliseconds [0-5, try 0.5]: ")
            noise_level = float(noise_input) if noise_input.strip() else 0.0
            if noise_level < 0:
                print("Noise level must be non-negative.")
                continue
            break
        except ValueError:
            print("Invalid number. Please enter a numeric value.")
    
    print(f"\nActual receiver position: {actual_receiver}")
    print(f"Noise level: {noise_level} ms")
    print()
    
    # Simulate signal transmission and measurement
    print("=" * 30)
    print("SIGNAL TRANSMISSION")
    print("=" * 30)
    
    # Calculate time-of-flight with noise
    tof1 = localizer.time_of_flight(tx1_pos, actual_receiver, noise_level)
    tof2 = localizer.time_of_flight(tx2_pos, actual_receiver, noise_level)
    
    print(f"Signal from TX1 arrives after: {tof1*1000:.3f} ms")
    print(f"Signal from TX2 arrives after: {tof2*1000:.3f} ms")
    print(f"Measured distance to TX1: {tof1 * localizer.speed_of_sound:.2f} m")
    print(f"Measured distance to TX2: {tof2 * localizer.speed_of_sound:.2f} m")
    
    # Perform localization
    print("\n" + "=" * 30)
    print("LOCALIZATION CALCULATION")
    print("=" * 30)
    
    try:
        estimated_receiver = localizer.localize(tx1_pos, tx2_pos, tof1, tof2)
        
        # Calculate accuracy
        error = localizer.distance_between_points(actual_receiver, estimated_receiver)
        
        print(f"Estimated position: ({estimated_receiver[0]:.2f}, {estimated_receiver[1]:.2f})")
        print(f"True position:      {actual_receiver}")
        print(f"Localization error: {error:.2f} m")
        
        # Create visualization
        print("\nGenerating visualization...")
        localizer.visualize(tx1_pos, tx2_pos, actual_receiver, estimated_receiver, tof1, tof2)
        
        # Explain results
        print("\n" + "=" * 30)
        print("EXPLANATION")
        print("=" * 30)
        print("The receiver is located at the intersection of two circles:")
        print(f"• Circle 1: centered at TX1 with radius {tof1 * localizer.speed_of_sound:.1f}m")
        print(f"• Circle 2: centered at TX2 with radius {tof2 * localizer.speed_of_sound:.1f}m")
        
        if noise_level > 0:
            print(f"\nWith {noise_level}ms noise, the timing uncertainty is ±{noise_level}ms,")
            print(f"which corresponds to distance uncertainty of ±{noise_level/1000 * localizer.speed_of_sound:.2f}m")
        
        if error < 1.0:
            print(f"\n✓ Excellent localization! Error is only {error:.2f}m")
        elif error < 5.0:
            print(f"\n~ Good localization. Error is {error:.2f}m")
        else:
            print(f"\n⚠ Poor localization. Error is {error:.2f}m - try reducing noise")
            
    except Exception as e:
        print(f"Localization failed: {e}")
        print("This can happen with high noise levels or poor transmitter geometry")

def noise_comparison():
    """Compare localization accuracy across different noise levels"""
    print("\n" + "=" * 50)
    print("NOISE IMPACT ANALYSIS")
    print("=" * 50)
    
    localizer = TOFLocalizer()
    tx1_pos = (20, 20)
    tx2_pos = (80, 80)
    actual_receiver = (50, 60)
    
    noise_levels = [0, 0.1, 0.3, 0.5, 1.0, 2.0]  # milliseconds
    trials_per_level = 100
    
    results = []
    
    print("Running analysis with 100 trials per noise level...")
    print("Noise (ms) | Mean Error (m) | Std Dev (m) | Success Rate")
    print("-" * 55)
    
    for noise in noise_levels:
        errors = []
        successes = 0
        
        for _ in range(trials_per_level):
            try:
                tof1 = localizer.time_of_flight(tx1_pos, actual_receiver, noise)
                tof2 = localizer.time_of_flight(tx2_pos, actual_receiver, noise)
                estimated = localizer.localize(tx1_pos, tx2_pos, tof1, tof2)
                error = localizer.distance_between_points(actual_receiver, estimated)
                
                if error < 100:  # Filter out obviously wrong solutions
                    errors.append(error)
                    successes += 1
            except:
                pass
        
        if errors:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            success_rate = successes / trials_per_level
            results.append((noise, mean_error, std_error, success_rate))
            print(f"{noise:8.1f}   | {mean_error:10.2f}   | {std_error:9.2f}   | {success_rate:8.1%}")
        else:
            print(f"{noise:8.1f}   | Failed to localize with this noise level")
    
    # Plot results
    if len(results) > 1:
        noise_vals, mean_errors, std_errors, success_rates = zip(*results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Error plot
        ax1.errorbar(noise_vals, mean_errors, yerr=std_errors, 
                    marker='o', capsize=5, linewidth=2, markersize=8)
        ax1.set_xlabel('Noise Level (ms)')
        ax1.set_ylabel('Localization Error (m)')
        ax1.set_title('Error vs Noise Level')
        ax1.grid(True, alpha=0.3)
        
        # Success rate plot
        ax2.plot(noise_vals, [s*100 for s in success_rates], 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Noise Level (ms)')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate vs Noise Level')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main program"""
    print("Welcome to the Time-of-Flight Localization Simulator!")
    print("\nThis program demonstrates how a receiver can determine its position")
    print("by measuring signal arrival times from two known transmitters.")
    
    while True:
        print("\nOptions:")
        print("1. Run single localization demo")
        print("2. Run noise impact analysis")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            run_simulation()
        elif choice == '2':
            noise_comparison()
        elif choice == '3':
            print("Thanks for using the TOF Localization Simulator!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    # Set random seed for reproducible results (optional)
    np.random.seed(42)
    main()