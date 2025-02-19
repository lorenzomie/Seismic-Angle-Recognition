import sys
sys.path.append("/home/EU/chmielel/ws/src/signals/Seismic-Angle-Recognition/seismic_angle_recognition/src") 

from pathlib import Path
import numpy as np
from typing import List, Tuple
from axitra import Axitra, moment
import convmPy
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

def create_sources(center_lat: float, center_lon: float, center_depth: float, distance: float, distance_perc: float, N: int) -> Tuple[np.ndarray, List[float], List[float]]:
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
    - A list of angles for each source
    - A list of distances from the center for each source
    """
    # 1 degree of latitude is approximately 111,320 meters
    lat_degree_to_meters = 111320
    lon_degree_to_meters = 111320  # Approximate for small areas (it changes based on latitude)
    
    # Generate N sources equidistant from the center
    sources = []
    angles = []
    distances = []
    for i in range(N):
        # Calculate the angle for each source (equally spaced around a circle)
        angle = 2 * np.pi * i / N  # Angle in radians
        
        # Vary the distance for each source
        varied_distance = distance * (1 + distance_perc * np.random.randn())
        
        # Calculate the change in lat/lon based on the varied distance
        delta_lat = varied_distance / lat_degree_to_meters
        delta_lon = varied_distance / lon_degree_to_meters
        
        # Calculate the offsets in latitude and longitude based on the angle
        offset_lat = delta_lat * np.cos(angle)
        offset_lon = delta_lon * np.sin(angle)
        
        # Create the source as a numpy array: [index, lat, lon, depth]
        source = np.array([i + 1, center_lat + offset_lat, center_lon + offset_lon, center_depth])
        angles.append(angle / np.pi * 180)
        distances.append(varied_distance)
        sources.append(source)
    
    return np.array(sources), angles, distances

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

def remove_temp_files(directory: Path):
    """
    Remove all .stat, .hist, .res, .source, and .data files in the given directory.
    
    Parameters:
    - directory: Path to the directory where the files should be removed
    """
    for ext in ['*.stat', '*.hist', '*.res', '*.source', '*.data']:
        for file in directory.glob(ext):
            try:
                file.unlink()
                print(f"Removed file: {file}")
            except Exception as e:
                print(f"Error removing file {file}: {e}")

def process_sources(sources, angles, distances, vel_model, station, std_hist, f_max, duration, axitra_path, noise_level):
    all_signals = []
    for source, angle, distance in zip(sources, angles, distances):
        source = np.reshape(source, (-1, 1)).T
        ap = Axitra(vel_model, station, source, fmax=f_max, duration=duration, xl=0., latlon=True, axpath=axitra_path)
        green_fn = moment.green(ap)
        t, sx, sy, sz = moment.conv(ap, std_hist, source_type=1, t0=2, unit=1)  # 1 should be Ricker
        print(len(t))
        sx += noise_level * np.random.randn(len(sx)) * sx
        sy += noise_level * np.random.randn(len(sy)) * sy
        sz += noise_level * np.random.randn(len(sz)) * sz
        signals = np.stack((sx, sy, sz), axis=-1)
        
        signal_data = {
            'signals': signals,
            'angle': angle,
            'distance': distance
        }
        all_signals.append(signal_data)
    return all_signals

@hydra.main(config_path="./config/", config_name="config.yaml")
def main(cfg: DictConfig):

    cfg = cfg.data_preparation

    center_lat = cfg.center_lat
    center_lon = cfg.center_lon
    center_depth = cfg.center_depth
    distance = cfg.distance
    distance_perc = cfg.distance_perc
    duration = cfg.duration
    f_max = cfg.f_max
    N = cfg.N
    
    axitra_path = cfg.axitra_path
    figures_path = Path(cfg.figures_path)
    sources, angles, distances = create_sources(center_lat, center_lon, center_depth, distance, distance_perc, N)
    station = create_station(1, center_lat, center_lon, center_depth + 100)
    std_hist = create_history()
    vel_model = create_velocity_model()
    noise_level = cfg.noise_level

    all_signals = process_sources(sources, angles, distances, vel_model, station, std_hist, f_max, duration, axitra_path, noise_level)
    
    # Create additional 20 samples of sources with distance_perc=0 and N=20
    additional_sources, additional_angles, additional_distances = create_sources(center_lat, center_lon, center_depth, distance, 0, 20)
    additional_signals = process_sources(additional_sources, additional_angles, additional_distances, vel_model, station, std_hist, f_max, duration, axitra_path, noise_level)
    
    # np.save(Path(cfg.output_data_path) / Path(cfg.output_file), all_signals)
    print(f"Saved signals to {Path(cfg.output_data_path) / Path(cfg.output_file)}")

    np.save(Path(cfg.output_data_path) / Path("additional_signals.npy"), additional_signals)
    print(f"Saved additional signals to {Path(cfg.output_data_path) / Path('additional_signals.npy')}")

    temp_files_directory = Path(cfg.temp_files_directory)
    remove_temp_files(temp_files_directory)

if __name__ == "__main__":
    main()
