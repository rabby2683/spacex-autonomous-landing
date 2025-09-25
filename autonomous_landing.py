import numpy as np
import math
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

class LandingPhase(Enum):
    INITIAL_DESCENT = 1
    GUIDANCE = 2
    FINAL_APPROACH = 3
    TOUCHDOWN = 4

@dataclass
class VehicleState:
    position: np.ndarray  # [x, y, z] meters
    velocity: np.ndarray  # [vx, vy, vz] m/s
    attitude: np.ndarray  # [roll, pitch, yaw] radians
    mass: float           # kg
    throttle: float       # 0-1

class AutonomousLandingSystem:
    def __init__(self, target_landing_zone: np.ndarray):
        self.target_zone = target_landing_zone
        self.phase = LandingPhase.INITIAL_DESCENT
        self.telemetry_data = []
        self.prediction_horizon = 5.0  # seconds
        
        # PID Controllers for each axis
        self.pid_gains = {
            'x': {'P': 0.8, 'I': 0.1, 'D': 0.3},
            'y': {'P': 0.8, 'I': 0.1, 'D': 0.3},
            'z': {'P': 1.2, 'I': 0.2, 'D': 0.4}
        }
        self.integral_errors = np.zeros(3)
        self.previous_errors = np.zeros(3)
        
    def calculate_trajectory_correction(self, 
                                      current_state: VehicleState,
                                      wind_conditions: np.ndarray,
                                      sensor_data: Dict) -> Tuple[np.ndarray, float]:
        """
        Calculate optimal trajectory correction using model predictive control
        """
        # Predict future state
        predicted_states = self.predict_trajectory(current_state, wind_conditions)
        
        # Calculate errors
        position_error = self.target_zone - current_state.position
        velocity_error = -current_state.velocity  # Target velocity is 0 at landing
        
        # Phase-based control logic
        if self.phase == LandingPhase.INITIAL_DESCENT:
            thrust_vector = self._initial_descent_control(position_error, velocity_error)
        elif self.phase == LandingPhase.GUIDANCE:
            thrust_vector = self._guidance_control(position_error, velocity_error, sensor_data)
        elif self.phase == LandingPhase.FINAL_APPROACH:
            thrust_vector = self._final_approach_control(position_error, velocity_error)
        else:
            thrust_vector = self._touchdown_control(position_error, velocity_error)
        
        # Update landing phase
        self._update_landing_phase(current_state)
        
        return thrust_vector, self._calculate_optimal_throttle(thrust_vector, current_state.mass)
    
    def predict_trajectory(self, 
                          current_state: VehicleState,
                          wind_conditions: np.ndarray,
                          time_steps: int = 50) -> List[VehicleState]:
        """
        Predict future trajectory using physics model
        """
        dt = 0.1  # time step
        predicted_states = []
        state = current_state
        
        for _ in range(time_steps):
            # Simple physics model (can be replaced with more complex dynamics)
            acceleration = self._calculate_acceleration(state, wind_conditions)
            
            # Update velocity and position
            new_velocity = state.velocity + acceleration * dt
            new_position = state.position + new_velocity * dt
            
            # Create new state
            new_state = VehicleState(
                position=new_position,
                velocity=new_velocity,
                attitude=state.attitude,
                mass=state.mass * 0.999,  # Fuel consumption
                throttle=state.throttle
            )
            
            predicted_states.append(new_state)
            state = new_state
        
        return predicted_states
    
    def _calculate_acceleration(self, state: VehicleState, wind: np.ndarray) -> np.ndarray:
        """Calculate acceleration based on current state and environmental factors"""
        gravity = np.array([0, 0, -9.81])  # m/s²
        thrust_acceleration = state.throttle * 20 * np.array([0, 0, 1])  # Max thrust 20 m/s²
        drag_acceleration = -0.1 * (state.velocity - wind)  # Simple drag model
        
        return gravity + thrust_acceleration + drag_acceleration
    
    def _initial_descent_control(self, pos_error: np.ndarray, vel_error: np.ndarray) -> np.ndarray:
        """Control logic for initial descent phase"""
        # Prioritize altitude control with gentle horizontal adjustments
        kp = np.array([0.5, 0.5, 1.0])  # Position gains
        kd = np.array([0.3, 0.3, 0.5])  # Velocity gains
        
        return kp * pos_error + kd * vel_error
    
    def _guidance_control(self, pos_error: np.ndarray, vel_error: np.ndarray, 
                         sensor_data: Dict) -> np.ndarray:
        """Advanced guidance control with sensor fusion"""
        # Sensor fusion from GPS, IMU, and vision systems
        fused_position = self._sensor_fusion(sensor_data)
        
        # Adaptive gains based on altitude
        altitude = fused_position[2]
        adaptive_gain = 1.0 / max(altitude, 1.0)
        
        kp = np.array([0.8, 0.8, 1.2]) * adaptive_gain
        kd = np.array([0.5, 0.5, 0.8]) * adaptive_gain
        
        return kp * pos_error + kd * vel_error
    
    def _sensor_fusion(self, sensor_data: Dict) -> np.ndarray:
        """Fuse data from multiple sensors for better position estimation"""
        gps_data = sensor_data.get('gps', np.zeros(3))
        imu_data = sensor_data.get('imu', np.zeros(3))
        vision_data = sensor_data.get('vision', np.zeros(3))
        
        # Simple weighted average (can be replaced with Kalman filter)
        weights = np.array([0.6, 0.3, 0.1])  # GPS most reliable
        sensors = np.vstack([gps_data, imu_data, vision_data])
        
        return np.average(sensors, axis=0, weights=weights)
    
    def _update_landing_phase(self, current_state: VehicleState):
        """Update landing phase based on current state"""
        altitude = current_state.position[2]
        horizontal_error = np.linalg.norm(current_state.position[:2] - self.target_zone[:2])
        
        if altitude > 1000:
            self.phase = LandingPhase.INITIAL_DESCENT
        elif altitude > 100:
            self.phase = LandingPhase.GUIDANCE
        elif altitude > 10:
            self.phase = LandingPhase.FINAL_APPROACH
        else:
            self.phase = LandingPhase.TOUCHDOWN
    
    def _calculate_optimal_throttle(self, thrust_vector: np.ndarray, mass: float) -> float:
        """Calculate optimal throttle setting"""
        required_thrust = np.linalg.norm(thrust_vector) * mass
        max_thrust = mass * 20  # 20 m/s² acceleration capability
        
        return np.clip(required_thrust / max_thrust, 0, 1)

# Simulation and Testing
class LandingSimulation:
    def __init__(self):
        self.landing_system = AutonomousLandingSystem(
            target_landing_zone=np.array([0, 0, 0])
        )
        
    def run_simulation(self, initial_altitude: float = 2000):
        """Run complete landing simulation"""
        # Initial vehicle state
        vehicle = VehicleState(
            position=np.array([500, 300, initial_altitude]),
            velocity=np.array([50, 30, -80]),
            attitude=np.array([0, 0, 0]),
            mass=10000,  # 10 tons
            throttle=0.0
        )
        
        simulation_data = []
        time_steps = 0
        max_steps = 1000
        
        while vehicle.position[2] > 0.1 and time_steps < max_steps:
            # Simulate sensor data with some noise
            sensor_data = {
                'gps': vehicle.position + np.random.normal(0, 1, 3),
                'imu': vehicle.velocity + np.random.normal(0, 0.1, 3),
                'vision': vehicle.position + np.random.normal(0, 0.5, 3)
            }
            
            # Wind conditions
            wind = np.array([np.random.normal(0, 5), 
                           np.random.normal(0, 5), 
                           0])
            
            # Get control commands from autonomous system
            thrust_vector, throttle = self.landing_system.calculate_trajectory_correction(
                vehicle, wind, sensor_data
            )
            
            # Update vehicle state
            vehicle.throttle = throttle
            acceleration = self.landing_system._calculate_acceleration(vehicle, wind)
            vehicle.velocity += acceleration * 0.1
            vehicle.position += vehicle.velocity * 0.1
            vehicle.mass *= 0.999  # Fuel consumption
            
            # Store telemetry
            simulation_data.append({
                'time': time_steps * 0.1,
                'position': vehicle.position.copy(),
                'velocity': vehicle.velocity.copy(),
                'throttle': throttle,
                'phase': self.landing_system.phase
            })
            
            time_steps += 1
        
        return simulation_data
    
    def plot_results(self, simulation_data):
        """Plot simulation results"""
        times = [data['time'] for data in simulation_data]
        positions = np.array([data['position'] for data in simulation_data])
        throttles = [data['throttle'] for data in simulation_data]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 3D Trajectory
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title('3D Landing Trajectory')
        
        # Altitude vs Time
        ax2.plot(times, positions[:, 2])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Altitude (m)')
        ax2.set_title('Altitude Profile')
        ax2.grid(True)
        
        # Horizontal Position
        ax3.plot(positions[:, 0], positions[:, 1])
        ax3.plot([0], [0], 'ro', markersize=10, label='Target')
        ax3.set_xlabel('X Position (m)')
        ax3.set_ylabel('Y Position (m)')
        ax3.set_title('Horizontal Trajectory')
        ax3.legend()
        ax3.grid(True)
        
        # Throttle Control
        ax4.plot(times, throttles)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Throttle (%)')
        ax4.set_title('Engine Throttle Control')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

# Run the simulation
if __name__ == "__main__":
    simulator = LandingSimulation()
    results = simulator.run_simulation(initial_altitude=2000)
    simulator.plot_results(results)
    
    # Analyze landing precision
    final_position = results[-1]['position']
    landing_error = np.linalg.norm(final_position[:2])
    print(f"Landing Error: {landing_error:.2f} meters")
    print(f"Final Velocity: {np.linalg.norm(results[-1]['velocity']):.2f} m/s")