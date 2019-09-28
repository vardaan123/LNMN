import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams.update({'font.size': 12})
plt.gcf().subplots_adjust(left=0.15)

def get_entropy_coeff(epoch_id):
    if epoch_id <= 19:
        return (1.0 - 2.0*epoch_id/49.0)
    else:
        return (11.0/49.0 - 6.0/49.0 * (epoch_id - 19.0))

sns.set_style("darkgrid")
x = list(range(30))
y = [get_entropy_coeff(item) for item in x]


plt.plot(x, y)
plt.xlabel('Epoch ID')
plt.ylabel(r'$\lambda_w$')
plt.savefig('entropy_coeff_plot.png')
