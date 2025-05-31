import numpy as np
import traceback
import logging
import time
from joblib import load

log = logging.getLogger(__name__)

# Single muscle class which handles the distance to pressure conversion
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


NBR_MUSCLES = 6
#class converts muscle lengths to pressure using ANN platform_muscle class. 
class DistanceToPressure:
    def __init__(self, max_compression, max_length):
        self.muscles = [Muscle() for _ in range(NBR_MUSCLES)]
        self.max_muscle_lengths = np.full(NBR_MUSCLES, max_length)
        self.last_frame_time = None
        self.load = 24

    def load_data(self, file_path):
        #Load ANN model and set this for each muscle  
        log.info(f"Loading model from: {file_path}")  
        model = load(file_path)
        for muscle in self.muscles:
            muscle.set_model(model)

    def set_load(self, load):
        self.load = load
        for muscle in self.muscles:
            muscle.load = load

    def muscle_length_to_pressure(self, muscle_lengths):
        muscle_lengths = np.asarray(muscle_lengths, dtype=int)
        if muscle_lengths.shape != self.max_muscle_lengths.shape:
            raise ValueError("Invalid number of muscle lengths")
        muscle_compressions =  self.max_muscle_lengths - muscle_lengths
        return self.muscle_compression_to_pressure(muscle_compressions)
        
    def muscle_compression_to_pressure(self, compressions):
        now = time.perf_counter()
        if self.last_frame_time is None:
            self.last_frame_time = now
            return [0] * len(self.muscles)

        dt = now - self.last_frame_time
        self.last_frame_time = now

        return [
            muscle.get_pressure(compression, dt)
            for muscle, compression in zip(self.muscles, compressions)
        ]

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

