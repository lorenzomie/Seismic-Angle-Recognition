# HOWTO

## Clone the Repository

To clone the repository, run the following command:

```bash
git clone https://github.com/lorenzomie/Seismic-Angle-Recognition.git
cd Seismic-Angle-Recognition
```

## Set Up the Linux Environment
Ensure you have Python 3.8 or higher installed.

## Create and Activate a Virtual Environment
To create a virtual environment, run the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Python Requirements
With the virtual environment activated, install the Python requirements listed in the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Install Axitra locally
To install Axitra locally, navigate to the [source directory](seismic_angle_recognition/src) and run the installation commands:

```bash
cd seismic_angle_recognition/src
```

For detailed information about axitra refers to [Axitra README](seismic_angle_recognition/src/README.md)

## Run the Scripts

To create the signals, run the following command:

```bash
python3 seismic_angle_recognition/create_signals.py
```

To train the model, run the following command:

```bash
python3 seismic_angle_recognition/train_model.py
```

All hyperparameters can be set in the [config.yaml](seismic_angle_recognition/config/config.yaml) file located in the config directory. 

### Additionals

If PROTOCOL_BUFFER is giving error please use:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

If you want to track metrics with mlflow, please forward a local port in another terminal:
```bash
python3 -m mlflow server --host 127.0.0.1 --port 8080
```