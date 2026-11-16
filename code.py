import pickle
import joblib
import time
from pathlib import Path

import numpy as np
import serial
import threading


EMG_WEIGHT = 0.4
IMU_WEIGHT = 0.6
FUSION_THRESHOLD = 0.5
FEATURE_DIM = 40


def load_models(emg_model_path="emg_model.pkl", imu_model_path="svm_model.pkl"):
    """Load pre-trained EMG and IMU models from joblib/pickle files."""
    emg_path = Path(emg_model_path)
    imu_path = Path(imu_model_path)

    if not emg_path.exists():
        raise FileNotFoundError(f"EMG model not found: {emg_path}")
    if not imu_path.exists():
        raise FileNotFoundError(f"IMU model not found: {imu_path}")

    # Load EMG model (joblib)
    emg_model = joblib.load(emg_path)

    # Load IMU model (joblib SVM)
    imu_model = joblib.load(imu_path)

    return emg_model, imu_model


class ArduinoInterface:
    """Handles serial communication with Arduino for motor control."""

    def __init__(self, port="COM3", baudrate=9600, timeout=1.0):
        """
        Initialize Arduino serial connection.
        port: COM port (Windows) or /dev/ttyUSB* (Linux)
        baudrate: Serial speed (default 9600 for Arduino)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.is_connected = False

    def connect(self):
        """Establish serial connection to Arduino."""
        try:
            self.ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout
            )
            self.is_connected = True
            print(f"[ARDUINO] Connected to {self.port} at {self.baudrate} baud")
            time.sleep(2)  # Arduino bootloader delay
            return True
        except serial.SerialException as e:
            print(f"[ARDUINO] Failed to connect: {e}")
            self.is_connected = False
            return False

    def send_motor_signal(self, signal):
        """
        Send motor control signal to Arduino.
        signal: 1 (activate motor/tremor) or 0 (deactivate)
        """
        if not self.is_connected:
            print("[ARDUINO] Not connected. Skipping send.")
            return False

        try:
            msg = f"{int(signal)}\n"
            self.ser.write(msg.encode())
            return True
        except serial.SerialException as e:
            print(f"[ARDUINO] Send failed: {e}")
            return False

    def receive_ack(self):
        """Wait for acknowledgment from Arduino."""
        if not self.is_connected:
            return None

        try:
            if self.ser.in_waiting > 0:
                ack = self.ser.readline().decode().strip()
                return ack
        except serial.SerialException as e:
            print(f"[ARDUINO] Receive failed: {e}")
        return None

    def disconnect(self):
        """Close serial connection."""
        if self.ser and self.is_connected:
            self.ser.close()
            self.is_connected = False
            print("[ARDUINO] Disconnected")


def _to_feature_vector(features, expected_dim=FEATURE_DIM):
    """
    Convert input to a 2D feature vector with shape (1, expected_dim).
    Accepts a 1D feature vector or a 2D row vector.
    """
    arr = np.asarray(features, dtype=np.float64)

    if arr.ndim == 1:
        if arr.size != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} features, got {arr.size}."
            )
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError("Feature matrix must have shape (1, N) for real-time inference.")
        if arr.shape[1] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} features, got {arr.shape[1]}."
            )
    else:
        raise ValueError("Feature input must be 1D or 2D.")

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_emg_features(emg_input):
    """
    Prepare EMG feature vector for inference.
    emg_input should be a real-time 40-dimensional feature vector.
    """
    return _to_feature_vector(emg_input, expected_dim=FEATURE_DIM)


def extract_imu_features(imu_input):
    """
    Prepare IMU feature vector for inference.
    imu_input should be a real-time 40-dimensional feature vector.
    """
    return _to_feature_vector(imu_input, expected_dim=FEATURE_DIM)


def predict_emg(emg_model, emg_features):
    """Run EMG model inference and return binary prediction (0 or 1)."""
    pred = int(emg_model.predict(emg_features)[0])
    return 1 if pred == 1 else 0


def predict_imu(imu_model, imu_features):
    """Run IMU model inference and return binary prediction (0 or 1)."""
    pred = int(imu_model.predict(imu_features), arduino_port="COM3", use_arduino=False):
    """
    Real-time inference entry point.
    interval_sec default 20 ms for low-latency wearable operation.

    Example usage for Arduino deployment:
        main(loop_forever=True, interval_sec=0.05, arduino_port="COM3", use_arduino=True)
    """
    emg_model, imu_model = load_models()

    arduino = None
    if use_arduino:
        arduino = ArduinoInterface(port=arduino_port)
        if not arduino.connect():
            print("[WARNING] Arduino not available. Running inference only.")
            arduino = None

    try:
        while True:
            emg_input = get_realtime_emg_features()
            imu_input = get_realtime_imu_features()

            fusion_result, motor_signal = run_inference_once(
                emg_model=emg_model,
                imu_model=imu_model,
                emg_input=emg_input,
                imu_input=imu_input,
            )

            print(
                f"EMG={fusion_result['emg_pred']} IMU={fusion_result['imu_pred']} "
                f"score={fusion_result['final_score']:.2f} "
                f"status={fusion_result['status']} motor_signal={motor_signal}"
            )

            # Send to Arduino if connected
            if arduino and arduino.is_connected:
                arduino.send_motor_signal(motor_signal)
                ack = arduino.receive_ack()
                if ack:
                    print(f"[ARDUINO ACK] {ack}")

            if not loop_forever:
                break

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping inference...")
    finally:
        if arduino:
            arduino.disconnect(nal_score),
        "final_decision": tremor_detected,
        "status": status,
    }


def motor_signal_from_decision(final_decision):
    """Map final decision to Arduino motor control signal."""
    return 1 if int(final_decision) == 1 else 0


def run_inference_once(emg_model, imu_model, emg_input, imu_input):
    """Single-cycle inference suitable for low-latency streaming loops."""
    emg_features = extract_emg_features(emg_input)
    imu_features = extract_imu_features(imu_input)

    emg_pred = predict_emg(emg_model, emg_features)
    imu_pred = predict_imu(imu_model, imu_features)

    fusion_result = fuse_predictions(emg_pred, imu_pred)
    motor_signal = motor_signal_from_decision(fusion_result["final_decision"])

    return fusion_result, motor_signal


def get_realtime_emg_features():
    """
    Placeholder for real sensor pipeline.
    Replace this with EMG acquisition + preprocessing code.
    """
    return np.zeros(FEATURE_DIM, dtype=np.float64)


def get_realtime_imu_features():
    """
    Placeholder for real sensor pipeline.
    Replace this with IMU acquisition + preprocessing code.
    """
    return np.zeros(FEATURE_DIM, dtype=np.float64)


def main(loop_forever=False, interval_sec=0.02):
    """
    Real-time inference entry point.
    interval_sec default 20 ms for low-latency wearable operation.
    """
    emg_model, imu_model = load_models()

    while True:
        emg_input = get_realtime_emg_features()
        imu_input = get_realtime_imu_features()

        fusion_result, motor_signal = run_inference_once(
            emg_model=emg_model,
            imu_model=imu_model,
            emg_input=emg_input,
            imu_input=imu_input,
        )

        print(
            f"EMG={fusion_result['emg_pred']} IMU={fusion_result['imu_pred']} "
            f"score={fusion_result['final_score']:.2f} "
            f"status={fusion_result['status']} motor_signal={motor_signal}"
        )

        # TODO: send motor_signal to Arduino serial, e.g. ser.write(f"{motor_signal}\n".encode())

        if not loop_forever:
            break

        time.sleep(interval_sec)


if __name__ == "__main__":
    main(loop_forever=False)