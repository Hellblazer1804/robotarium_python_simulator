import matplotlib.pyplot as plt
import pandas as pd
file1 = pd.read_csv('type.csv')
file2 = pd.read_csv('mobility.csv')
file3 = pd.read_csv('quality.csv')
file4 = pd.read_csv('range.csv')

plt.figure(figsize=(12,8))
plt.plot(file1, label='Locational Cost', color='b')
plt.plot(file2, label='Temporal Cost', color='g')
plt.plot(file3, label='Power Cost', color='r')
plt.plot(file4,label='Range Limited Cost', color='y')

# Adding labels and title
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Line Plot for Baselines')
plt.legend()
plt.grid(True)

plt.show()