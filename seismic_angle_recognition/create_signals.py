import sys
sys.path.append(".") 

import os
import numpy as np
from typing import List
from axitra import Axitra, moment
import convmPy
import matplotlib.pyplot as plt

def create_sources(center_lat: float, center_lon: float, center_depth: float, distance: float, N: int) -> List[np.ndarray]:
    """
    Create N sources equidistant from a central point, distributed along a circle.
    
    Parameters:
    - center_lat: Latitude of the central point
    - center_lon: Longitude of the central point
    - center_depth: Depth of the central point (in meters)
    - distance: Distance from the center to each source (in meters)
    - N: Number of sources to generate
    
    Returns:
    - A list of N sources, each represented as a numpy array [index, lat, lon, depth]
    """
    # 1 degree of latitude is approximately 111,320 meters
    lat_degree_to_meters = 111320
    lon_degree_to_meters = 111320  # Approximate for small areas (it changes based on latitude)
    
    # Calculate the change in lat/lon based on the distance
    delta_lat = distance / lat_degree_to_meters
    delta_lon = distance / lon_degree_to_meters
    
    # Generate N sources equidistant from the center
    sources = []
    for i in range(N):
        # Calculate the angle for each source (equally spaced around a circle)
        angle = 2 * np.pi * i / N  # Angle in radians
        
        # Calculate the offsets in latitude and longitude based on the angle
        offset_lat = delta_lat * np.cos(angle)
        offset_lon = delta_lon * np.sin(angle)
        
        # Create the source as a numpy array: [index, lat, lon, depth]
        source = np.array([i + 1, center_lat + offset_lat, center_lon + offset_lon, center_depth])
        sources.append(source)
    
    return np.array(sources)

def create_station(index: int, lat: float, lon: float, depth: float) -> np.ndarray:
    """
    Create a station with the given parameters.
    
    Parameters:
    - index: Index of the station
    - lat: Latitude of the station
    - lon: Longitude of the station
    - depth: Depth of the station
    """
    
    station = np.array([index, lat, lon, depth])
    station = station.reshape(1, -1)
    return station

def create_velocity_model() -> np.ndarray:
    """Copied form example.ipynb"""
    return np.array([[1000., 5000., 2886., 2700., 1000., 500.],
                  [0., 6000., 3886., 2700., 1000., 500.]])

def create_history()-> np.ndarray:
    return np.array([[1, 7.5e20, 148.0, 84.0, -47.0, 0., 0., 10.0]])

def main():
    center_lat = 45.0 
    center_lon = 2.0  
    center_depth = 1000.0 
    distance = 1000.0
    N = 10
    axitra_path = ""
    
    sources = create_sources(center_lat, center_lon, center_depth, distance, N)
    station = create_station(1, center_lat, center_lon, center_depth+100)
    std_hist = create_history()
    vel_model = create_velocity_model()
    
    # Print the sources
    for source in sources:
        print(station)
        ap = Axitra(vel_model, station, sources, fmax=1.0, duration=50., xl=0., latlon=True, axpath=axitra_path)
        green_fn = moment.green(ap)
        t, sx, sy, sz = moment.conv(ap,std_hist,source_type=1,t0=2,unit=1) # 1 should be Ricker
        plt.figure(figsize=(18, 9))
        ier=plt.plot(t,sx[1,:],t,sx[2,:],t,sx[3,:],t,sx[4,:],t,sx[0,:],)
        
        

if __name__ == "__main__":
    main()
