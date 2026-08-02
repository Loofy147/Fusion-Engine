"""
Comprehensive verification test suite for the FusionEngine v4.4 Peripheral Systems.
Tests all 7 modules:
  1. TimeSyncBuffer
  2. SchemaAdapter
  3. ValidationLayer
  4. SensorHealth
  5. DriftMonitor
  6. AlertManager
  7. SafetyInterlock
  8. FusionSystemV4_4 End-to-End Pipeline
"""

import numpy as np
import warnings
from fusion_engine_v4_4 import FusionConfig
from fusion_system_v4_4 import (
    TimeSyncBuffer,
    SchemaAdapter,
    ValidationLayer,
    SensorHealth,
    DriftMonitor,
    AlertManager,
    SafetyInterlock,
    FusionSystemV4_4
)

results = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    results.append((status, name))
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


# ==============================================================================
# 1. TimeSyncBuffer Tests
# ==============================================================================
def test_time_sync_buffer():
    # M=3, K=2
    sync = TimeSyncBuffer(num_sensors_m=3, dim_k=2, max_sync_delay=0.5, sync_mode="sample_and_hold")

    # Ingest data out of order
    sync.ingest(sensor_id=0, timestamp=1.0, data=np.array([10.0, 20.0]))
    sync.ingest(sensor_id=0, timestamp=0.5, data=np.array([5.0, 5.0]))

    # Check sorting
    check("TimeSyncBuffer: internal sorting of out-of-order inputs",
          sync.buffers[0][0][0] == 0.5 and sync.buffers[0][1][0] == 1.0)

    # Ingest other sensors
    sync.ingest(sensor_id=1, timestamp=0.9, data=np.array([11.0, 21.0]))
    sync.ingest(sensor_id=2, timestamp=1.1, data=np.array([12.0, 22.0]))

    # Retrieve frame at t=1.0 under Sample-And-Hold
    # Sensor 0: t=1.0 -> exact match [10, 20]
    # Sensor 1: t=0.9 -> latest t <= 1.0 within 0.5 delay -> [11, 21]
    # Sensor 2: t=1.1 is future, no t <= 1.0 -> NaN
    matrix, mask = sync.get_synchronized_frame(1.0)
    check("TimeSyncBuffer: sample_and_hold logic",
          np.allclose(matrix[0], [10.0, 20.0]) and
          np.allclose(matrix[1], [11.0, 21.0]) and
          np.isnan(matrix[2][0]) and
          mask[0] == True and mask[1] == True and mask[2] == False)

    # Nearest Neighbor Mode
    sync_nearest = TimeSyncBuffer(num_sensors_m=3, dim_k=2, max_sync_delay=0.5, sync_mode="nearest")
    sync_nearest.ingest(0, 0.5, np.array([5.0, 5.0]))
    sync_nearest.ingest(1, 0.9, np.array([11.0, 21.0]))
    sync_nearest.ingest(2, 1.1, np.array([12.0, 22.0]))

    matrix_nn, mask_nn = sync_nearest.get_synchronized_frame(1.0)
    # Sensor 0: t=0.5 is 0.5 away -> valid
    # Sensor 1: t=0.9 is 0.1 away -> valid [11, 21]
    # Sensor 2: t=1.1 is 0.1 away -> valid [12, 22]
    check("TimeSyncBuffer: nearest neighbor alignment",
          mask_nn[0] == True and mask_nn[1] == True and mask_nn[2] == True and
          np.allclose(matrix_nn[1], [11.0, 21.0]) and np.allclose(matrix_nn[2], [12.0, 22.0]))

    # Interpolate Mode
    sync_interp = TimeSyncBuffer(num_sensors_m=3, dim_k=2, max_sync_delay=0.5, sync_mode="interpolate")
    sync_interp.ingest(0, 0.0, np.array([0.0, 0.0]))
    sync_interp.ingest(0, 1.0, np.array([10.0, 20.0]))

    matrix_ip, mask_ip = sync_interp.get_synchronized_frame(0.4)
    # Sensor 0 should interpolate 40% between 0 and 10 -> [4.0, 8.0]
    check("TimeSyncBuffer: linear interpolation alignment",
          mask_ip[0] == True and np.allclose(matrix_ip[0], [4.0, 8.0]))


# ==============================================================================
# 2. SchemaAdapter Tests
# ==============================================================================
def test_schema_adapter():
    adapter = SchemaAdapter(num_sensors_m=3, dim_k=2)

    # Sensor 0: Scale and offset config
    adapter.configure_sensor(sensor_id=0, scale=2.0, bias=np.array([1.0, -1.0]))
    v0 = np.array([5.0, 5.0])
    adapted0 = adapter.adapt(0, v0)  # 5.0 * 2.0 + 1 = 11.0; 5.0 * 2.0 - 1 = 9.0
    check("SchemaAdapter: scale and bias transformation", np.allclose(adapted0, [11.0, 9.0]))

    # Sensor 1: dimension mapping mapping 3D raw input to 2D standard
    adapter.configure_sensor(sensor_id=1, dim_map=[0, 2])  # extract index 0 and 2
    v1 = np.array([100.0, 200.0, 300.0])
    adapted1 = adapter.adapt(1, v1)
    check("SchemaAdapter: dimension subset mapping", np.allclose(adapted1, [100.0, 300.0]))

    # Sensor 2: coordinate rotation (90 deg counter-clockwise)
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    adapter.configure_sensor(sensor_id=2, rotation=rot)
    v2 = np.array([1.0, 0.0])
    adapted2 = adapter.adapt(2, v2)  # [0, 1]
    check("SchemaAdapter: coordinate rotation matrix", np.allclose(adapted2, [0.0, 1.0]))


# ==============================================================================
# 3. ValidationLayer Tests
# ==============================================================================
def test_validation_layer():
    val = ValidationLayer(
        num_sensors_m=3,
        dim_k=2,
        lower_limits=np.array([0.0, -10.0]),
        upper_limits=np.array([100.0, 10.0]),
        max_rate_of_change=5.0
    )

    # Frame 1: nominal
    x1 = np.array([
        [50.0, 0.0],
        [10.0, -5.0],
        [90.0, 5.0]
    ])
    mat1, mask1 = val.validate(x1)
    check("ValidationLayer: nominal inputs valid", np.all(mask1) and np.allclose(mat1, x1))

    # Frame 2: Out of limits, NaN, and rate of change check
    # Sensor 0: out of physical limits (y=15.0 exceeds upper limit 10.0)
    # Sensor 1: contains np.nan
    # Sensor 2: jumps from [90, 5] to [10, 5], jump of 80 in x dimension exceeds max_rate_of_change=5.0
    x2 = np.array([
        [50.0, 15.0],
        [10.0, np.nan],
        [10.0, 5.0]
    ])
    mat2, mask2 = val.validate(x2)
    check("ValidationLayer: bounds, NaN, and rate of change validation",
          mask2[0] == False and mask2[1] == False and mask2[2] == False and
          np.all(np.isnan(mat2)))


# ==============================================================================
# 4. SensorHealth Tests
# ==============================================================================
def test_sensor_health():
    health = SensorHealth(num_sensors_m=3, window_size=10)

    # Sensor 0 is completely healthy
    for _ in range(10):
        health.update_metrics(sensor_id=0, trust=0.9, elevated=False, valid=True)
    check("SensorHealth: HEALTHY classification", health.get_health_status(0) == "HEALTHY")

    # Sensor 1 has high drop rate
    for _ in range(5):
        health.update_metrics(sensor_id=1, trust=0.8, elevated=False, valid=True)
    for _ in range(5):
        health.update_metrics(sensor_id=1, trust=0.0, elevated=False, valid=False)
    check("SensorHealth: FAULTY classification due to high drop rate", health.get_health_status(1) == "FAULTY")

    # Sensor 2 has moderate trust and intermittent elevation -> DEGRADED
    for _ in range(10):
        health.update_metrics(sensor_id=2, trust=0.5, elevated=True if _ % 3 == 0 else False, valid=True)
    check("SensorHealth: DEGRADED classification", health.get_health_status(2) == "DEGRADED")


# ==============================================================================
# 5. DriftMonitor Tests
# ==============================================================================
def test_drift_monitor():
    monitor = DriftMonitor(num_sensors_m=3, dim_k=2, slack_g=0.5, threshold_h=3.0)

    # Nominal zero-mean residues
    for _ in range(5):
        alert = monitor.update_and_detect(sensor_id=0, normalized_residual=np.array([0.1, -0.1]))

    check("DriftMonitor: no alert for small zero-mean residues", alert is None)

    # Inject systematic small drift (1.0 residual, exceeds slack_g=0.5, building CUSUM)
    alert = None
    for i in range(10):
        alert = monitor.update_and_detect(sensor_id=0, normalized_residual=np.array([1.0, 0.0]))
        if alert is not None:
            break

    check("DriftMonitor: CUSUM cumulative drift detection triggered",
          alert is not None and alert["sensor_id"] == 0 and alert["triggered_dimensions"] == [0],
          f"triggered at step {i} with message: {alert['message'] if alert else ''}")


# ==============================================================================
# 6. AlertManager Tests
# ==============================================================================
def test_alert_manager():
    manager = AlertManager(alert_cooldown_steps=5)

    events_received = []
    def log_event(event_type, msg, meta):
        events_received.append((event_type, meta.get("sensor_id")))

    manager.subscribe("sensor_fault", log_event)

    # First fault publish: should succeed
    d1 = manager.publish("sensor_fault", "sensor faulty", {"sensor_id": 0})
    # Second immediate fault publish on same sensor: should be debounced
    d2 = manager.publish("sensor_fault", "sensor faulty again", {"sensor_id": 0})
    # Fault publish on a DIFFERENT sensor: should succeed
    d3 = manager.publish("sensor_fault", "other sensor faulty", {"sensor_id": 1})

    check("AlertManager: subscription and rate-limiting debouncing",
          d1 == True and d2 == False and d3 == True and len(events_received) == 2,
          f"Events logged: {events_received}")

    # Advance clock beyond cooldown steps (cooldown=5)
    for _ in range(6):
        manager.step()

    # Now publishing on sensor 0 again should succeed
    d4 = manager.publish("sensor_fault", "sensor 0 faulty after cooldown", {"sensor_id": 0})
    check("AlertManager: cooldown release", d4 == True and len(events_received) == 3)


# ==============================================================================
# 7. SafetyInterlock Tests
# ==============================================================================
def test_safety_interlock():
    # Simple dead reckoning model: x_next = x_curr + 1.0
    def sample_process_model(state, dt):
        return state + dt * 1.0

    interlock = SafetyInterlock(
        dim_k=2,
        num_sensors_m=3,
        process_model_fn=sample_process_model,
        max_failsafe_steps=3
    )

    # Nominal update
    status1, state1 = interlock.process(fused_state=np.array([10.0, 10.0]), engine_status={"input_rejected": False})
    check("SafetyInterlock: NORMAL state", status1 == "NORMAL" and np.allclose(state1, [10.0, 10.0]))

    # Step 1 of Input Rejection -> Active failsafe, uses process model
    status2, state2 = interlock.process(fused_state=None, engine_status={"input_rejected": True}, dt=2.0)
    check("SafetyInterlock: FAILSAFE_ACTIVE dead-reckoning",
          status2 == "FAILSAFE_ACTIVE" and np.allclose(state2, [12.0, 12.0]))

    # Step 2 & 3 of Input Rejection -> Exceeds max_failsafe_steps=3 -> TRIP
    _ = interlock.process(fused_state=None, engine_status={"input_rejected": True})
    status4, state4 = interlock.process(fused_state=None, engine_status={"input_rejected": True})
    check("SafetyInterlock: TRIP threshold triggers fallback holding last safe state",
          status4 == "TRIP" and np.allclose(state4, [13.0, 13.0]))


# ==============================================================================
# 8. FusionSystemV4_4 End-to-End Pipeline
# ==============================================================================
def test_fusion_system_e2e():
    cfg = FusionConfig(dim_k=2, num_sensors_m=3, calib_window=5)

    system = FusionSystemV4_4(
        engine_config=cfg,
        sync_mode="nearest",
        max_sync_delay=0.1,
        cusum_g=0.5,
        cusum_h=2.0,
        alert_cooldown=5,
        max_failsafe_steps=4
    )

    # Register alert callbacks
    alerts_fired = []
    def record_alert(ev_type, msg, meta):
        alerts_fired.append((ev_type, meta.get("sensor_id"), msg))

    system.alert_manager.subscribe("sensor_fault", record_alert)
    system.alert_manager.subscribe("sensor_drift", record_alert)
    system.alert_manager.subscribe("total_observation_loss", record_alert)
    system.alert_manager.subscribe("safety_interlock_trip", record_alert)

    # Configure adapter rules for sensor 1 (needs scaling / offset)
    # standard = raw * 2.0 - 1.0
    system.adapter.configure_sensor(sensor_id=1, scale=2.0, bias=-1.0)

    # 1. Feed calibration phase readings
    for i in range(5):
        t = i * 0.1
        # Ingest sensor 0
        system.ingest_sensor_reading(0, t, np.array([0.0, 0.0]))
        # Ingest sensor 1 (raw is 0.5, standard = 0.5*2 - 1 = 0)
        system.ingest_sensor_reading(1, t, np.array([0.5, 0.5]))
        # Ingest sensor 2
        system.ingest_sensor_reading(2, t, np.array([0.0, 0.0]))

        out = system.update_pipeline(t)

    check("E2E Pipeline: calibrated successfully",
          out["engine_output"]["calibrating"] == False and system.engine.sigma_reliable is not None)

    # 2. Feed nominal tracking steps
    for i in range(5, 10):
        t = i * 0.1
        system.ingest_sensor_reading(0, t, np.array([1.0, 1.0]))
        system.ingest_sensor_reading(1, t, np.array([1.0, 1.0]))  # raw 1.0, adapted = 1.0*2 - 1 = 1.0
        system.ingest_sensor_reading(2, t, np.array([1.0, 1.0]))
        out = system.update_pipeline(t)

    check("E2E Pipeline: tracks state under nominal conditions", np.allclose(out["state"], [1.0, 1.0], atol=0.15))

    # 3. Sensor 2 starts drifting slowly
    # At target 2.0, sensor 2 reports increasingly shifted readings
    for i in range(10, 20):
        t = i * 0.1
        system.ingest_sensor_reading(0, t, np.array([2.0, 2.0]))
        system.ingest_sensor_reading(1, t, np.array([1.5, 1.5]))  # raw 1.5, adapted = 2.0
        # Slowly drift sensor 2: +0.6 per step
        drift_offset = 2.0 + (i - 9) * 0.4
        system.ingest_sensor_reading(2, t, np.array([drift_offset, drift_offset]))
        out = system.update_pipeline(t)

    # Check that drift is detected and alert fired
    drift_alerts = [a for a in alerts_fired if a[0] == "sensor_drift"]
    check("E2E Pipeline: drift alert fired for sensor 2", len(drift_alerts) > 0, f"Drift alerts: {drift_alerts}")

    # 4. Total dropouts leading to interlock trip
    # Ingest only NaNs for 5 consecutive steps
    out = None
    for i in range(20, 25):
        t = i * 0.1
        out = system.update_pipeline(t)

    check("E2E Pipeline: safety interlock trips on prolonged dropout",
          out is not None and out["interlock_status"] == "TRIP")

    # Verify alerting registry
    trip_alerts = [a for a in alerts_fired if a[0] == "safety_interlock_trip"]
    check("E2E Pipeline: safety trip alert logged", len(trip_alerts) > 0)


# ==============================================================================
# Execute Verification Suite
# ==============================================================================
if __name__ == "__main__":
    print("Executing FusionEngine v4.4 Expanded System Tests...\n")
    test_time_sync_buffer()
    test_schema_adapter()
    test_validation_layer()
    test_sensor_health()
    test_drift_monitor()
    test_alert_manager()
    test_safety_interlock()
    test_fusion_system_e2e()

    print(f"\nSYSTEM TEST SUMMARY: {sum(1 for s, _ in results if s == 'PASS')}/{len(results)} PASSED")
