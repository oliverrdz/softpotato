import matplotlib.pyplot as plt

import softpotato as sp

lsv = sp.lsv(-0.5, 0.5, 0.01, 1)
cv = sp.cv(0, 0.5, 1, 0.01, -0.5, 2)

plt.figure(1)
plt.plot(lsv[:, 1], lsv[:, 0])
plt.plot(cv[:, 1], cv[:, 0])
plt.xlabel("t / s")
plt.ylabel("E / V")
plt.grid()
plt.show()
