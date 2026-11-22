# Custom Waveforms & Segments

When an experiment does not fit a standard voltammetry or pulse template—such as conditioning holds followed by AC voltammetry or exponential relaxation—you can compose arbitrary excitation signals using `CustomWaveform` and `WaveformSegment` building blocks.

`CustomWaveform` automatically computes total `duration`, stitches segment boundaries as `event_times` for piecewise integration, and supports both `"potentiostatic"` and `"galvanostatic"` control modes.

## Available Segment Primitives

| Segment Class | Mathematical Form ($\tau \in [0, \Delta t]$) | Key Parameters |
| :--- | :--- | :--- |
| `ConstantSegment` | $y(\tau) = y_0$ | `value`, `duration` |
| `LinearRampSegment` | $y(\tau) = y_{\text{start}} + \frac{y_{\text{end}} - y_{\text{start}}}{\Delta t}\tau$ | `start`, `end`, `duration` (or `rate`) |
| `SinusoidalSegment` | $y(\tau) = y_0 + A \sin(2\pi f \tau + \phi)$ | `offset`, `amplitude`, `frequency`, `duration`, `phase` |
| `CallableSegment` | $y(\tau) = f(\tau)$ | `func`, `duration`, `event_times` (optional) |

### `CustomWaveform` Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `segments` | `Sequence[WaveformSegment]` | Non-empty ordered sequence of segment objects. |
| `mode` | `'potentiostatic'` or `'galvanostatic'` | Control mode (`'potentiostatic'` applies potential in $\text{V}$; `'galvanostatic'` applies current in $\text{A}$). |
| `n_points` | `int`, optional | Default number of evaluation points when `time_grid()` is called without arguments (default `1001`). |

## Full Runnable Script

```python
--8<-- "docs/scripts/guide_custom_waveform.py"
```
