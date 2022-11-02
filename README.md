# Script version of Soft Potato, the electrochemistry simulator.

For more info, visit [Soft Potato](https://softpotato.xyz). 
Preprint available at [ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/635c42f7cf6de97b4726accf)

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
