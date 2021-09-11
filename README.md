# Script version of Soft Potato, the electrochemistry simulator.

For more info, visit [Soft Potato](https://oliverrdz.xyz/soft-potato). To see Jupyter Notebook examples, visit [Examples](https://github.com/oliverrdz/softpotato_examples).

## Install
``` python
pip3 install softpotato

```

## Usage
``` python
from softpotato import *
import softpotato as sp

disc = sp.calc.MicroDisc(a=1e-4)
iLim = disc.iLim
print(iLim)

```
