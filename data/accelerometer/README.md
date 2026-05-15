# Accelerometer Data

This folder contains raw and processed accelerometer data for the ANN-from-scratch project.

## Raw Data

Raw logs should be placed here:

```text
raw/
|-- normal/
|   `-- teraterm.log
`-- anomaly/
    `-- teraterm.log
```

Raw files are Tera Term logs. Each line contains one timestamp followed by many accelerometer triples:

```text
[2026-05-14 20:06:52.009] x1 y1 z1 x2 y2 z2 ...
```

Keep raw data unchanged.

## Processed Data

Generated files go here:

```text
processed/
|-- normal_samples.csv
|-- anomaly_samples.csv
|-- window_features.csv
|-- train_window_features.csv
|-- test_window_features.csv
`-- plots/
```

## Labels

```text
0 = normal
1 = anomaly
```

## Units

Raw values are treated as `mg`.

Processed sample files include both:

```text
x_mg, y_mg, z_mg
x_g, y_g, z_g
```

where:

```text
g = mg / 1000
```

## Timing

Sensor rate:

```text
416 Hz
```

The most reliable time axis for machine learning is:

```text
time_seconds = sample_index / 416
```

Tera Term line timestamps are kept for traceability, but they are not used as the main ML time axis.
