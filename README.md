# Script version of Soft Potato, the electrochemistry simulator.

For more info, visit [Soft Potato](https://softpotato.xyz). 
Preprint available at [10.26434/chemrxiv-2022-4bs3w](https://10.26434/chemrxiv-2022-4bs3w)

## Install
``` python
pip install softpotato

```

## Usage
``` python
import softpotato as sp

disc = sp.calc.MicroDisc(a=1e-4)
iLim = disc.iLim
print(iLim)

```
