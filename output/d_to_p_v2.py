import numpy as np
import traceback
import logging
import time
from joblib import load

log = logging.getLogger(__name__)

# Single muscle class which handles the pressure each muscle should have
class Muscle:
    def __init__(self, smoothing_factor=0.3):
        self.model, self.pressure_min, self.pressure_max = None, None, None
        self.smoothing_factor = smoothing_factor
        self.previous_pressure = 0
        self.load = 24 #Load is 24kg by default
    
    def get_pressure(self, distance, dt):
        if(self.model == None):
            raise ValueError("Model is not loaded")

        # Calculate the velocity
        if hasattr(self, "last_distance"):
            velocity = (distance - self.last_distance) / dt
        else:
            velocity = 0

        current_time = time.perf_counter()

        # Prepare the input features: [c_t, c_t-1, velocity]
        """
        The ANN's input features to work out the muscle's appropiate pressure are:
        distance: Target distance the muscle has to reach
        last_distance: Previous distance that the muscle was asked to reach in the last frame
        velocity: The rate of change between distance and last_distance ie:(distance-last_distance)/dt
        load: the load that the muscle is set to be lifting
        """
        features = np.array([[distance, self.last_distance if hasattr(self, "last_distance") else 0, velocity, self.load]])
        
        # Make the prediction. Model gives scaled prediction hence pressure_max and pressure_min usage
        predicted_pressure = (self.model.predict(features)[0]* (self.pressure_max - self.pressure_min)) + self.pressure_min
        #print(time.perf_counter() - current_time)

        # Apply smoothing
        smoothed_pressure = self.smoothing_factor * predicted_pressure + (1 - self.smoothing_factor) * self.previous_pressure
        self.previous_pressure = smoothed_pressure
        
        # Update last distance
        self.last_distance = distance
        
        return int(smoothed_pressure)
    
    def set_model(self, model):
        self.model, self.pressure_min, self.pressure_max = model


# Class that wraps the 6 muscles into a single class from which the pressures can be obtained
class Platform_Muscles:
    def __init__(self):
        self.muscles = [Muscle() for _ in range(6)]
        self.load = 24
        self.last_frame_time = None

    def distance_to_pressures(self, distances):
        if(self.last_frame_time == None):
            self.last_frame_time = time.perf_counter()
            return 0

        current_time = time.perf_counter()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        print(dt)

        return [max(0, min(6000, muscle.get_pressure(distance, dt)))
                for muscle, distance in zip(self.muscles, distances)]
    
    def set_payload_per_muscle(self, nload):
        self.load = nload
        for muscle in self.muscles:
            muscle.load = self.load
        print(self.muscles[0].load)


#Shell class that implements my ANN platform_muscle class. This allows my code to be seamlesslyintegrated into the rest of the SimOpConsole.
class DistanceToPressure:
    def __init__(self, nbr_columns, max_length):
        self.nbr_columns = nbr_columns
        self.platform = Platform_Muscles()
        self.platform.set_payload_per_muscle(24)
        self.max_muscle_lengths = np.full(6, max_length, dtype=int)
        
        
    def load_data(self, file_path):
        #Load data with ANN loads ANN model
        print(file_path)
        model = load(file_path)
        for muscle in self.platform.muscles:
            muscle.set_model(model)
            
    def set_load(self, load):
        self.platform.set_payload_per_muscle(load/6)

    def muscle_length_to_pressure(self, muscle_lengths):
        muscle_lengths = np.asarray(muscle_lengths, dtype=int)
        if muscle_lengths.shape != self.max_muscle_lengths.shape:
            raise ValueError("Invalid number of muscle lengths")
        muscle_compressions =  self.max_muscle_lengths - muscle_lengths
        return self.muscle_compression_to_pressure(muscle_compressions)
    
    def muscle_compression_to_pressure(self, compressions):
       return self.platform.distance_to_pressures(compressions)

# ---------------------------------------------------------
# Test harness
# ---------------------------------------------------------
if __name__ == '__main__':
    max_range = 250
    max_length = 1000
    d2p = DistanceToPressure(max_range+1, max_length)
    if d2p.load_data("wheelchair_DtoP.csv"):
        d2p.set_load(24)  # Set test load

    print(d2p.d_to_p) 
    while True:
        user_input = input("Enter 6 lengths separated by commas or 'l,<load>' to set load (empty input to quit): ").strip()

        if not user_input:
            break

        if user_input.lower().startswith('l,'):
            try:
                load = int(user_input.split(',')[1])
                d2p.set_load(load) 
                print(f"Load updated to: {load}kg")
            except (IndexError, ValueError):
                print("Invalid load format. Use l,<number> (e.g., l,35)")
            continue

        try:
            lengths = [int(val) for val in user_input.split(',')]
            if len(lengths) != 6:
                print("Please enter exactly 6 numbers.")
                continue

            pressures = d2p.muscle_length_to_pressure(lengths)
            print(f"Input lengths: {lengths}")
            print(f"Resulting pressures: {pressures}")

        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

