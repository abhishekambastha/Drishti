import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

mu, sigma = 00, 1
x = mu + sigma*np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=True )

print n.shape
# add a 'best fit' line
#y = mlab.normpdf(bins, mu, sigma)
print bins.shape
print bins.dtype
print "Bins", bins
#l = plt.plot(bins, bins )

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=0,\ \sigma=1$')
plt.grid(True)
plt.show()
