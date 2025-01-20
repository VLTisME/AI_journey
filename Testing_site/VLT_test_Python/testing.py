import numpy as np
import matplotlib.pyplot as plt # type: ignore

x = np.array([1, 2, 3, 4, 6, 7])
y = np.array([3, 4, 5, 6, 10, 2])

plt.plot(x, y)
plt.show()