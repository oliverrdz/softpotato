# Waveforms (Potential Programs)

SoftPotato waveforms use a canonical NumPy array format with shape (n, 2)
where the columns are [E, t] (potential in V, time in s).

## Building blocks

- uniform_time_grid(...)
- lsv(...)
- cv(...)
- step(...)
- waveform_from_arrays(E, t)

