"""
FusionEngine v4.4 -- Peripheral Systems.
Implements the peripheral architecture modules for production deployment:
  1. TimeSyncBuffer: Temporal alignment of multi-rate streams.
  2. SchemaAdapter: Normalizes units, feature ordering, coordinate systems, and dimensions.
  3. ValidationLayer: Range checks, missing-data masking, and finite-value verification.
  4. SensorHealth: Diagnostic metrics and health status tracking.
  5. DriftMonitor: CUSUM-based slow drift detection.
  6. AlertManager: Debounced notification and alert routing.
  7. SafetyInterlock: Emergency fail-safe and fallback execution.

This module works alongside FusionEngineV4_4 to create a robust, end-to-end state estimation pipeline.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Literal, Callable
import numpy as np
import collections
import warnings

# Import the core engine and config
from fusion_engine_v4_4 import FusionEngineV4_4, FusionConfig


class TimeSyncBuffer:
    """
    TimeSyncBuffer aligns multi-rate, asynchronous, or out-of-order sensor streams
    into synchronized M x K state matrices.

    Supports three synchronization policies:
      - 'nearest': Finds the reading closest to the query timestamp within max_sync_delay.
      - 'sample_and_hold': Uses the most recent reading up to the query timestamp,
                           not older than max_sync_delay.
      - 'interpolate': Performs linear interpolation between the two closest readings
                       bounding the query timestamp, if both are within max_sync_delay.
    """
    def __init__(self,
                 num_sensors_m: int,
                 dim_k: int,
                 max_sync_delay: float = 0.5,
                 sync_mode: Literal["nearest", "sample_and_hold", "interpolate"] = "sample_and_hold",
                 max_buffer_len: int = 100):
        self.num_sensors_m = num_sensors_m
        self.dim_k = dim_k
        self.max_sync_delay = max_sync_delay
        self.sync_mode = sync_mode
        self.max_buffer_len = max_buffer_len

        # Deque for each sensor containing tuples of (timestamp, data_vector)
        self.buffers: Dict[int, collections.deque] = {
            i: collections.deque(maxlen=max_buffer_len) for i in range(num_sensors_m)
        }

    def ingest(self, sensor_id: int, timestamp: float, data: np.ndarray) -> None:
        """
        Ingests a reading from a specific sensor.

        Args:
            sensor_id: Index of the sensor (0 to M-1).
            timestamp: Event timestamp.
            data: State vector observation.
        """
        if not (0 <= sensor_id < self.num_sensors_m):
            raise ValueError(f"sensor_id={sensor_id} out of bounds for M={self.num_sensors_m}")

        vec = np.asarray(data, dtype=float)
        if vec.shape != (self.dim_k,):
            raise ValueError(f"Data shape {vec.shape} does not match dim_k={self.dim_k}")

        # Store in buffer and maintain sorted order by timestamp
        buf = self.buffers[sensor_id]
        buf.append((timestamp, vec))
        # Sort buffer by timestamp in case of out-of-order arrival
        sorted_items = sorted(buf, key=lambda x: x[0])
        buf.clear()
        buf.extend(sorted_items)

    def get_synchronized_frame(self, target_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts a synchronized frame of M x K observations for the given target timestamp.

        Returns:
            aligned_matrix: np.ndarray of shape (M, K), with np.nan for missing/invalid sensors.
            valid_mask: np.ndarray of shape (M,) indicating which sensors have valid, synchronized data.
        """
        aligned_matrix = np.full((self.num_sensors_m, self.dim_k), np.nan)
        valid_mask = np.zeros(self.num_sensors_m, dtype=bool)

        for sensor_id in range(self.num_sensors_m):
            buf = self.buffers[sensor_id]
            if not buf:
                continue

            times = np.array([item[0] for item in buf])
            vals = np.array([item[1] for item in buf])

            if self.sync_mode == "sample_and_hold":
                # Find the latest reading with t <= target_time
                idx = np.searchsorted(times, target_time, side="right") - 1
                if idx >= 0:
                    t_cand, val_cand = times[idx], vals[idx]
                    if abs(target_time - t_cand) <= self.max_sync_delay:
                        aligned_matrix[sensor_id] = val_cand
                        valid_mask[sensor_id] = True

            elif self.sync_mode == "nearest":
                # Find the reading with minimum absolute time difference
                diffs = np.abs(times - target_time)
                min_idx = np.argmin(diffs)
                if diffs[min_idx] <= self.max_sync_delay:
                    aligned_matrix[sensor_id] = vals[min_idx]
                    valid_mask[sensor_id] = True

            elif self.sync_mode == "interpolate":
                # If exact match exists
                exact_idx = np.where(np.isclose(times, target_time))[0]
                if exact_idx.size > 0:
                    aligned_matrix[sensor_id] = vals[exact_idx[0]]
                    valid_mask[sensor_id] = True
                    continue

                # Find bounding indices
                right_idx = np.searchsorted(times, target_time)
                left_idx = right_idx - 1

                if left_idx >= 0 and right_idx < len(times):
                    t_left, val_left = times[left_idx], vals[left_idx]
                    t_right, val_right = times[right_idx], vals[right_idx]

                    if (target_time - t_left <= self.max_sync_delay) and (t_right - target_time <= self.max_sync_delay):
                        # Perform linear interpolation
                        dt = t_right - t_left
                        if dt > 1e-9:
                            alpha = (target_time - t_left) / dt
                            aligned_matrix[sensor_id] = (1.0 - alpha) * val_left + alpha * val_right
                            valid_mask[sensor_id] = True
                        else:
                            aligned_matrix[sensor_id] = val_left
                            valid_mask[sensor_id] = True
                else:
                    # Fallback to nearest if we can't bound, but check delay
                    diffs = np.abs(times - target_time)
                    min_idx = np.argmin(diffs)
                    if diffs[min_idx] <= self.max_sync_delay:
                        aligned_matrix[sensor_id] = vals[min_idx]
                        valid_mask[sensor_id] = True

        return aligned_matrix, valid_mask


class SchemaAdapter:
    """
    SchemaAdapter normalizes units, feature ordering, coordinate systems, and dimensions
    for heterogeneous input streams.

    Transforms a raw sensor-specific vector into a standardized K-dimensional vector:
        x_standard = Rotation * ( (x_raw * Scale) + Bias ) [or projected using dim_map]
    """
    def __init__(self, num_sensors_m: int, dim_k: int):
        self.num_sensors_m = num_sensors_m
        self.dim_k = dim_k

        # Configurations per sensor
        self.scales: Dict[int, np.ndarray] = {}
        self.biases: Dict[int, np.ndarray] = {}
        self.dim_maps: Dict[int, List[int]] = {}
        self.rotations: Dict[int, np.ndarray] = {}

    def configure_sensor(self,
                         sensor_id: int,
                         scale: Optional[Union[float, np.ndarray]] = None,
                         bias: Optional[Union[float, np.ndarray]] = None,
                         dim_map: Optional[List[int]] = None,
                         rotation: Optional[np.ndarray] = None) -> None:
        """
        Configures conversion parameters for a given sensor.
        """
        if not (0 <= sensor_id < self.num_sensors_m):
            raise ValueError(f"sensor_id={sensor_id} out of bounds")

        if dim_map is not None:
            if len(dim_map) != self.dim_k:
                raise ValueError(f"dim_map length {len(dim_map)} must match target dim_k={self.dim_k}")
            self.dim_maps[sensor_id] = list(dim_map)

        # Determine raw dimensionality for scale/bias verification
        raw_dim = len(dim_map) if dim_map is not None else self.dim_k

        if scale is not None:
            if np.isscalar(scale):
                self.scales[sensor_id] = np.full(raw_dim, float(scale))
            else:
                scale_arr = np.asarray(scale, dtype=float)
                if scale_arr.shape != (raw_dim,):
                    raise ValueError(f"Scale shape {scale_arr.shape} must match raw_dim={raw_dim}")
                self.scales[sensor_id] = scale_arr

        if bias is not None:
            if np.isscalar(bias):
                self.biases[sensor_id] = np.full(raw_dim, float(bias))
            else:
                bias_arr = np.asarray(bias, dtype=float)
                if bias_arr.shape != (raw_dim,):
                    raise ValueError(f"Bias shape {bias_arr.shape} must match raw_dim={raw_dim}")
                self.biases[sensor_id] = bias_arr

        if rotation is not None:
            rot_arr = np.asarray(rotation, dtype=float)
            if rot_arr.shape != (self.dim_k, self.dim_k):
                raise ValueError(f"Rotation matrix shape {rot_arr.shape} must be ({self.dim_k}, {self.dim_k})")
            self.rotations[sensor_id] = rot_arr

    def adapt(self, sensor_id: int, raw_data: np.ndarray) -> np.ndarray:
        """
        Adapts raw heterogeneous sensor data into the standard K-dimensional space.
        """
        if not (0 <= sensor_id < self.num_sensors_m):
            raise ValueError(f"sensor_id={sensor_id} out of bounds")

        v = np.asarray(raw_data, dtype=float).copy()

        # 1. Apply dim_map if configured (projection/reordering)
        if sensor_id in self.dim_maps:
            mapping = self.dim_maps[sensor_id]
            # Ensure indices are within raw_data bounds
            if any(idx < 0 or idx >= len(v) for idx in mapping):
                raise ValueError(f"dim_map indices {mapping} exceed raw_data dimension {len(v)}")
            v_mapped = np.array([v[idx] for idx in mapping])
        else:
            if v.shape != (self.dim_k,):
                raise ValueError(f"Input shape {v.shape} does not match dim_k={self.dim_k}")
            v_mapped = v

        # 2. Scale & Bias (in the mapped dimension space)
        if sensor_id in self.scales:
            v_mapped = v_mapped * self.scales[sensor_id]
        if sensor_id in self.biases:
            v_mapped = v_mapped + self.biases[sensor_id]

        # 3. Rotation (in the final K-dimensional space)
        if sensor_id in self.rotations:
            v_mapped = self.rotations[sensor_id] @ v_mapped

        return v_mapped


class ValidationLayer:
    """
    ValidationLayer performs range checks, missing-data masking, and finite-value verification.
    Also implements derivative (rate-of-change) checks to reject sudden non-physical jumps.
    """
    def __init__(self,
                 num_sensors_m: int,
                 dim_k: int,
                 lower_limits: Optional[np.ndarray] = None,
                 upper_limits: Optional[np.ndarray] = None,
                 max_rate_of_change: Optional[Union[float, np.ndarray]] = None):
        self.num_sensors_m = num_sensors_m
        self.dim_k = dim_k
        self.lower_limits = np.asarray(lower_limits, dtype=float) if lower_limits is not None else None
        self.upper_limits = np.asarray(upper_limits, dtype=float) if upper_limits is not None else None

        if max_rate_of_change is not None:
            if np.isscalar(max_rate_of_change):
                self.max_rate_of_change = np.full(dim_k, float(max_rate_of_change))
            else:
                self.max_rate_of_change = np.asarray(max_rate_of_change, dtype=float)
        else:
            self.max_rate_of_change = None

        # Stores last validated values per sensor to compute rates of change
        self.last_valid_values: Dict[int, np.ndarray] = {}

    def validate(self, matrix_m_x_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validates the M x K observation matrix.
        Replaces invalid coordinates or entire rows with np.nan.

        Returns:
            validated_matrix: np.ndarray (M, K) where invalid entries are NaN.
            valid_sensor_mask: np.ndarray (M,) boolean mask indicating valid rows.
        """
        x = np.asarray(matrix_m_x_k, dtype=float).copy()
        if x.shape != (self.num_sensors_m, self.dim_k):
            raise ValueError(f"Matrix shape {x.shape} does not match ({self.num_sensors_m}, {self.dim_k})")

        valid_sensor_mask = np.ones(self.num_sensors_m, dtype=bool)

        for m in range(self.num_sensors_m):
            row = x[m]

            # 1. Finite check
            if not np.all(np.isfinite(row)):
                x[m] = np.nan
                valid_sensor_mask[m] = False
                continue

            # 2. Hard physical bounds range check
            if self.lower_limits is not None:
                if np.any(row < self.lower_limits):
                    x[m] = np.nan
                    valid_sensor_mask[m] = False
                    continue
            if self.upper_limits is not None:
                if np.any(row > self.upper_limits):
                    x[m] = np.nan
                    valid_sensor_mask[m] = False
                    continue

            # 3. Rate of change (derivative gate)
            if self.max_rate_of_change is not None and m in self.last_valid_values:
                prev_val = self.last_valid_values[m]
                diff = np.abs(row - prev_val)
                if np.any(diff > self.max_rate_of_change):
                    x[m] = np.nan
                    valid_sensor_mask[m] = False
                    continue

            # If still valid, update last_valid_values
            self.last_valid_values[m] = row.copy()

        return x, valid_sensor_mask


class SensorHealth:
    """
    SensorHealth tracks historical metrics (trust scores, elevation rates, and drop rates)
    in a sliding window and classifies each sensor's health status.

    Statuses:
      - 'HEALTHY': high trust, low elevation, low drops.
      - 'DEGRADED': moderate trust, intermittent elevation or drops.
      - 'FAULTY': very low trust, or highly persistent elevation or drop rates.
    """
    def __init__(self,
                 num_sensors_m: int,
                 window_size: int = 50,
                 healthy_trust: float = 0.7,
                 faulty_trust: float = 0.3,
                 healthy_elevation: float = 0.15,
                 faulty_elevation: float = 0.4,
                 healthy_drop: float = 0.1,
                 faulty_drop: float = 0.4):
        self.num_sensors_m = num_sensors_m
        self.window_size = window_size
        self.healthy_trust = healthy_trust
        self.faulty_trust = faulty_trust
        self.healthy_elevation = healthy_elevation
        self.faulty_elevation = faulty_elevation
        self.healthy_drop = healthy_drop
        self.faulty_drop = faulty_drop

        # Metrics lists/buffers per sensor
        self.trust_history: Dict[int, collections.deque] = {
            i: collections.deque(maxlen=window_size) for i in range(num_sensors_m)
        }
        self.elevation_history: Dict[int, collections.deque] = {
            i: collections.deque(maxlen=window_size) for i in range(num_sensors_m)
        }
        self.valid_history: Dict[int, collections.deque] = {
            i: collections.deque(maxlen=window_size) for i in range(num_sensors_m)
        }

    def update_metrics(self, sensor_id: int, trust: float, elevated: bool, valid: bool) -> str:
        """
        Updates the history for a sensor and returns its current health status.
        """
        if not (0 <= sensor_id < self.num_sensors_m):
            raise ValueError(f"sensor_id={sensor_id} out of bounds")

        self.trust_history[sensor_id].append(float(trust))
        self.elevation_history[sensor_id].append(bool(elevated))
        self.valid_history[sensor_id].append(bool(valid))

        return self.get_health_status(sensor_id)

    def get_health_status(self, sensor_id: int) -> str:
        """
        Computes the health status and returns a status string: 'HEALTHY', 'DEGRADED', or 'FAULTY'.
        """
        if not (0 <= sensor_id < self.num_sensors_m):
            raise ValueError(f"sensor_id={sensor_id} out of bounds")

        t_hist = self.trust_history[sensor_id]
        e_hist = self.elevation_history[sensor_id]
        v_hist = self.valid_history[sensor_id]

        if not t_hist:
            return "HEALTHY"  # Default startup status

        avg_trust = float(np.mean(t_hist))

        # Drop rate is the ratio of invalid frames to total window size
        drop_rate = 1.0 - (sum(v_hist) / len(v_hist))

        # Elevation rate is the ratio of elevated frames to valid frames in window
        valid_indices = [i for i, val in enumerate(v_hist) if val]
        if valid_indices:
            elevation_rate = sum(e_hist[i] for i in valid_indices) / len(valid_indices)
        else:
            elevation_rate = 1.0  # All frames dropped implies severe fault

        # Classification rules
        if avg_trust < self.faulty_trust or elevation_rate >= self.faulty_elevation or drop_rate >= self.faulty_drop:
            return "FAULTY"
        elif avg_trust >= self.healthy_trust and elevation_rate <= self.healthy_elevation and drop_rate <= self.healthy_drop:
            return "HEALTHY"
        else:
            return "DEGRADED"

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Returns diagnostic dictionary for all sensors.
        """
        diag = {}
        for i in range(self.num_sensors_m):
            t_hist = self.trust_history[i]
            v_hist = self.valid_history[i]
            e_hist = self.elevation_history[i]

            avg_trust = float(np.mean(t_hist)) if t_hist else 1.0
            drop_rate = 1.0 - (sum(v_hist) / len(v_hist)) if v_hist else 0.0

            valid_indices = [idx for idx, val in enumerate(v_hist) if val]
            elevation_rate = sum(e_hist[idx] for idx in valid_indices) / len(valid_indices) if valid_indices else 0.0

            diag[i] = {
                "status": self.get_health_status(i),
                "avg_trust": avg_trust,
                "drop_rate": drop_rate,
                "elevation_rate": elevation_rate,
            }
        return diag


class DriftMonitor:
    """
    DriftMonitor implements a CUSUM (Cumulative Sum) algorithm per sensor and state dimension
    to detect slow, sub-threshold, systematic drifts before they trigger instantaneous elevation.
    """
    def __init__(self,
                 num_sensors_m: int,
                 dim_k: int,
                 slack_g: float = 0.5,
                 threshold_h: float = 5.0):
        self.num_sensors_m = num_sensors_m
        self.dim_k = dim_k
        self.g = slack_g
        self.h = threshold_h

        # Running CUSUM statistics: shape (M, K)
        self.s_pos = np.zeros((num_sensors_m, dim_k), dtype=float)
        self.s_neg = np.zeros((num_sensors_m, dim_k), dtype=float)

    def update_and_detect(self, sensor_id: int, normalized_residual: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Updates CUSUM statistic for a sensor with its K-dimensional normalized residual
        (e.g., (x_m - belief) / sigma_reliable).

        Returns:
            alert_details: Dict if drift is detected in any dimension, else None.
        """
        if not (0 <= sensor_id < self.num_sensors_m):
            raise ValueError(f"sensor_id={sensor_id} out of bounds")

        z = np.asarray(normalized_residual, dtype=float)
        if z.shape != (self.dim_k,):
            raise ValueError(f"residual shape {z.shape} must match dim_k={self.dim_k}")

        # Update positive and negative CUSUM
        # S_pos = max(0, S_pos + z - g)
        # S_neg = max(0, S_neg - z - g)
        self.s_pos[sensor_id] = np.maximum(0.0, self.s_pos[sensor_id] + z - self.g)
        self.s_neg[sensor_id] = np.maximum(0.0, self.s_neg[sensor_id] - z - self.g)

        pos_triggered = self.s_pos[sensor_id] > self.h
        neg_triggered = self.s_neg[sensor_id] > self.h

        if np.any(pos_triggered) or np.any(neg_triggered):
            triggered_dims = np.where(pos_triggered | neg_triggered)[0].tolist()
            return {
                "sensor_id": sensor_id,
                "triggered_dimensions": triggered_dims,
                "s_pos": self.s_pos[sensor_id].copy(),
                "s_neg": self.s_neg[sensor_id].copy(),
                "message": f"Sensor {sensor_id} is drifting in dimensions {triggered_dims}"
            }
        return None

    def reset_sensor(self, sensor_id: int) -> None:
        """
        Resets running CUSUM statistics for a sensor (e.g. after maintenance or recalibration).
        """
        if not (0 <= sensor_id < self.num_sensors_m):
            raise ValueError(f"sensor_id={sensor_id} out of bounds")
        self.s_pos[sensor_id].fill(0.0)
        self.s_neg[sensor_id].fill(0.0)


class AlertManager:
    """
    AlertManager handles event registration, notification debouncing, and callback dispatching.
    Prevents alert storms by applying a cooldown (in timesteps) per alert key.
    """
    def __init__(self, alert_cooldown_steps: int = 30):
        self.alert_cooldown_steps = alert_cooldown_steps
        self.subscribers: Dict[str, List[Any]] = collections.defaultdict(list)
        # Tracks last timestep an alert key was published: Dict[alert_key, timestep]
        self.last_published_t: Dict[str, int] = {}
        self.t = 0

    def step(self) -> None:
        """
        Increments the alert manager's internal clock (timesteps).
        """
        self.t += 1

    def subscribe(self, event_type: str, callback: Any) -> None:
        """
        Subscribes a callback to an event type (e.g. 'sensor_fault', 'sensor_drift', 'prolonged_freeze').
        """
        self.subscribers[event_type].append(callback)

    def publish(self, event_type: str, message: str, metadata: Dict[str, Any]) -> bool:
        """
        Publishes an event if it is not debounced.

        Args:
            event_type: Event classification.
            message: Descriptive text.
            metadata: Structured diagnostic payload.

        Returns:
            dispatched: True if dispatched to listeners, False if debounced.
        """
        # Create a unique key for debouncing (e.g. event_type + sensor_id if present)
        sensor_id = metadata.get("sensor_id", "")
        alert_key = f"{event_type}_{sensor_id}"

        if alert_key in self.last_published_t:
            elapsed = self.t - self.last_published_t[alert_key]
            if elapsed < self.alert_cooldown_steps:
                return False  # Debounced

        self.last_published_t[alert_key] = self.t

        # Dispatch to subscribers
        for callback in self.subscribers[event_type]:
            try:
                callback(event_type, message, metadata)
            except Exception as e:
                warnings.warn(f"Alert subscriber callback failed: {e}", stacklevel=2)

        return True


class SafetyInterlock:
    """
    SafetyInterlock executes emergency fallback and mitigation behavior.

    States:
      - 'NORMAL': Nominal operation.
      - 'FAILSAFE_ACTIVE': Fallback engaged (e.g., dead reckoning on input rejection).
      - 'TRIP': Critical hazard / violation or prolonged failure. Outputs safe default, halts loops.
    """
    def __init__(self,
                 dim_k: int,
                 num_sensors_m: int,
                 process_model_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
                 max_failsafe_steps: int = 50,
                 custom_safety_check_fn: Optional[Callable[[np.ndarray], bool]] = None):
        self.dim_k = dim_k
        self.num_sensors_m = num_sensors_m
        self.process_model_fn = process_model_fn
        self.max_failsafe_steps = max_failsafe_steps
        self.custom_safety_check_fn = custom_safety_check_fn

        self.status = "NORMAL"
        self.last_safe_state: Optional[np.ndarray] = None
        self.failsafe_step_count = 0

    def process(self,
                fused_state: Optional[np.ndarray],
                engine_status: Dict[str, Any],
                dt: float = 1.0) -> Tuple[str, Optional[np.ndarray]]:
        """
        Enforces safety policies, updating the interlock state and generating the safe state output.
        """
        input_rejected = engine_status.get("input_rejected", False)
        is_frozen = engine_status.get("is_frozen", False)
        contiguous_frozen_steps = engine_status.get("contiguous_frozen_steps", 0)

        # 1. Evaluate custom safety rules first
        tripped_by_safety = False
        if fused_state is not None and self.custom_safety_check_fn is not None:
            if not self.custom_safety_check_fn(fused_state):
                tripped_by_safety = True

        # 2. State transition logic
        if self.status == "TRIP" or tripped_by_safety:
            self.status = "TRIP"
            # Output last known safe state (or zero vector if none exists)
            out_state = self.last_safe_state if self.last_safe_state is not None else np.zeros(self.dim_k)
            return self.status, out_state

        if input_rejected:
            self.status = "FAILSAFE_ACTIVE"
            self.failsafe_step_count += 1

            if self.failsafe_step_count >= self.max_failsafe_steps:
                self.status = "TRIP"
                out_state = self.last_safe_state if self.last_safe_state is not None else np.zeros(self.dim_k)
                return self.status, out_state

            # Perform dead-reckoning or hold last safe state
            if self.process_model_fn is not None and self.last_safe_state is not None:
                propagated = self.process_model_fn(self.last_safe_state, dt)
                self.last_safe_state = propagated
                out_state = propagated
            else:
                out_state = self.last_safe_state
        else:
            # We have a valid update!
            # If engine is frozen too long, trigger FAILSAFE or TRIP
            if is_frozen and contiguous_frozen_steps >= self.max_failsafe_steps:
                self.status = "TRIP"
                out_state = self.last_safe_state if self.last_safe_state is not None else np.zeros(self.dim_k)
                return self.status, out_state

            self.status = "NORMAL"
            self.failsafe_step_count = 0
            if fused_state is not None:
                self.last_safe_state = fused_state.copy()
            out_state = fused_state

        return self.status, out_state


class FusionSystemV4_4:
    """
    FusionSystemV4_4 is the complete production-grade state estimation pipeline.
    It coordinates ingestion, temporal alignment, schema standardization, range-based validation,
    CUSUM drift detection, bayesian core fusion, historical diagnostics, alerting, and interlock fallbacks.
    """
    def __init__(self,
                 engine_config: Optional[FusionConfig] = None,
                 max_sync_delay: float = 0.5,
                 sync_mode: Literal["nearest", "sample_and_hold", "interpolate"] = "sample_and_hold",
                 lower_limits: Optional[np.ndarray] = None,
                 upper_limits: Optional[np.ndarray] = None,
                 max_rate_of_change: Optional[Union[float, np.ndarray]] = None,
                 cusum_g: float = 0.5,
                 cusum_h: float = 5.0,
                 alert_cooldown: int = 30,
                 max_failsafe_steps: int = 50,
                 process_model_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
                 custom_safety_check_fn: Optional[Callable[[np.ndarray], bool]] = None):

        self.cfg = engine_config if engine_config is not None else FusionConfig()
        self.dim_k = self.cfg.dim_k
        self.num_sensors_m = self.cfg.num_sensors_m

        # Initialize modules
        self.engine = FusionEngineV4_4(self.cfg)
        self.sync_buffer = TimeSyncBuffer(self.num_sensors_m, self.dim_k, max_sync_delay, sync_mode)
        self.adapter = SchemaAdapter(self.num_sensors_m, self.dim_k)
        self.validation = ValidationLayer(self.num_sensors_m, self.dim_k, lower_limits, upper_limits, max_rate_of_change)
        self.health = SensorHealth(self.num_sensors_m, window_size=50)
        self.drift_monitor = DriftMonitor(self.num_sensors_m, self.dim_k, cusum_g, cusum_h)
        self.alert_manager = AlertManager(alert_cooldown)
        self.interlock = SafetyInterlock(self.dim_k, self.num_sensors_m, process_model_fn, max_failsafe_steps, custom_safety_check_fn)

        # Keep track of previous health statuses for transition alerts
        self.last_sensor_statuses: Dict[int, str] = {i: "HEALTHY" for i in range(self.num_sensors_m)}
        self.timestep = 0

    def ingest_sensor_reading(self, sensor_id: int, timestamp: float, raw_data: np.ndarray) -> None:
        """
        Ingests and adapts raw readings from multi-rate asynchronous sensors.
        """
        adapted_data = self.adapter.adapt(sensor_id, raw_data)
        self.sync_buffer.ingest(sensor_id, timestamp, adapted_data)

    def update_pipeline(self, timestamp: float, dt: float = 1.0) -> Dict[str, Any]:
        """
        Executes one full iteration of the state-fusion pipeline.

        Args:
            timestamp: Target synchronization time.
            dt: Timestep delta for dead reckoning.
        """
        self.timestep += 1
        self.alert_manager.step()

        # 1. Aligns asynchronous inputs
        aligned_matrix, sync_mask = self.sync_buffer.get_synchronized_frame(timestamp)

        # 2. Validates aligned inputs
        validated_matrix, valid_mask = self.validation.validate(aligned_matrix)

        # 3. Invokes core Bayesian and GNC/Median Fusion Engine
        engine_out = self.engine.update(validated_matrix)

        fused_raw_state = engine_out.get("state")
        sensor_trusts = engine_out.get("sensor_trust")
        is_frozen = engine_out.get("is_frozen", False)
        contiguous_frozen_steps = engine_out.get("contiguous_frozen_steps", 0)
        input_rejected = engine_out.get("input_rejected", False)

        # 4. Long-horizon sub-threshold drift detection (CUSUM)
        if not input_rejected and self.engine.sigma_reliable is not None:
            sigma_rel = self.engine.sigma_reliable
            for m in range(self.num_sensors_m):
                if valid_mask[m]:
                    # Residual normalized against operating noise
                    resid_vector = validated_matrix[m] - self.engine.belief
                    norm_resid = resid_vector / (sigma_rel + self.cfg.reg)
                    drift_alert = self.drift_monitor.update_and_detect(m, norm_resid)
                    if drift_alert is not None:
                        self.alert_manager.publish(
                            event_type="sensor_drift",
                            message=drift_alert["message"],
                            metadata={"sensor_id": m, "triggered_dimensions": drift_alert["triggered_dimensions"]}
                        )

        # 5. SensorHealth Monitoring
        for m in range(self.num_sensors_m):
            trust = sensor_trusts[m]

            # Reconstruct if elevated
            elevated = False
            if not input_rejected and self.engine.sigma_reliable is not None:
                thresh = (self.engine.cfg.detect_sigma_mult * self.engine.sigma_reliable) ** 2 * self.dim_k
                resid_sq = ((validated_matrix[m] - self.engine.belief) ** 2).sum()
                elevated = resid_sq > thresh

            current_status = self.health.update_metrics(m, trust, elevated, valid_mask[m])

            # Detect transitions to 'FAULTY' to raise alarm
            prev_status = self.last_sensor_statuses[m]
            if current_status == "FAULTY" and prev_status != "FAULTY":
                self.alert_manager.publish(
                    event_type="sensor_fault",
                    message=f"Sensor {m} status transitioned to FAULTY",
                    metadata={"sensor_id": m, "avg_trust": float(np.mean(self.health.trust_history[m]))}
                )
            self.last_sensor_statuses[m] = current_status

        # 6. Pipeline level alerts
        if is_frozen and contiguous_frozen_steps >= 15:
            self.alert_manager.publish(
                event_type="prolonged_freeze",
                message=f"System has been in frozen mode for {contiguous_frozen_steps} steps",
                metadata={"contiguous_frozen_steps": contiguous_frozen_steps}
            )

        if input_rejected:
            self.alert_manager.publish(
                event_type="total_observation_loss",
                message="All sensor inputs rejected: complete observation loss",
                metadata={"valid_sensor_mask": valid_mask.tolist()}
            )

        # 7. Safety Interlock checks and output generation
        interlock_status, final_state = self.interlock.process(fused_raw_state, engine_out, dt)

        if interlock_status != "NORMAL":
            self.alert_manager.publish(
                event_type="safety_interlock_trip",
                message=f"Safety interlock triggered: status={interlock_status}",
                metadata={"status": interlock_status, "failsafe_step_count": self.interlock.failsafe_step_count}
            )

        return {
            "state": final_state,
            "interlock_status": interlock_status,
            "engine_output": engine_out,
            "sensor_healths": {m: self.health.get_health_status(m) for m in range(self.num_sensors_m)},
            "diagnostics": self.health.get_diagnostics(),
            "valid_sensor_mask": valid_mask,
        }
