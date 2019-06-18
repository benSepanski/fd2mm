import matplotlib.pyplot as plt
data = open('out.txt', 'r')

data = [tuple(map(float, line.split())) for line in data]
x, y = [d[0] for d in data], [d[1] for d in data]

plt.scatter(x, y)
plt.xlabel("wave number")
plt.ylabel("(Ku, u)")
plt.show()
