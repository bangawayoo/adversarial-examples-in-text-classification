##%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')
plt.style.use(['ggplot', 'paper', 'paper_twocol'])



##% Figure
robust_spectrum = pd.read_csv('fig/data/spectrum_kPCA.csv', names=['max','min']).astype(float)
naive_spectrum = pd.read_csv('fig/data/spectrum_naive.csv', names=['max','min']).astype(float)

fig, ax = plt.subplots()
p = ax.scatter(robust_spectrum.iloc[:,1], robust_spectrum.iloc[:,0], label='kPCA')
ax.scatter(naive_spectrum.iloc[:,1], naive_spectrum.iloc[:,0], label='Naive')
ax.set_title('Max. vs. Min. Eigenvalues \nof Covariance Matrix', fontsize=28)
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_ylabel('Max. E.V', fontsize=20)
# ax.set_xlabel('Min. E.V', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=35)
ax.legend(loc='best', prop={'size':25})
fig.tight_layout()
plt.show()
fig.savefig('fig/spectrum.pdf', dpi=300)


##%
df = pd.read_csv('fig/data/curse_of_dim.csv')
steps = df['steps']

fig, ax = plt.subplots(figsize=(5,5))
ax2= ax.twinx()

color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)
p1, = ax.plot(steps, df['mu'], color=color1, label=r'$\mu$ Relative Error', linewidth=4)
p2, = ax2.plot(steps, df['cov'], color=color2, label=r'$\Sigma$ Relative Error', linewidth=4)
last_value = df['mu'].iloc[-1]
last_step = steps.iloc[-1]
ann = ax.annotate(f"{last_value:.2E}", xy=(last_step, last_value), xytext=(2**(12.2), last_value+0.05), fontsize=27,\
            arrowprops=dict(facecolor='black', shrink=0.05),xycoords='data')


lns = [p1, p2]
ax.legend(handles=lns, loc='best', prop={'size':25})
ax.set_xscale('log', base=2)

ax.set_title("Relative Error vs. Sample Size", fontsize=30)
ax.set_ylabel(r'$\frac{||\mu-\tilde\mu||}{||\mu||}$', rotation=90, fontsize=35)
ax2.set_ylabel(r'$\frac{||\Sigma-\tilde\Sigma||}{||\Sigma||}$', rotation=90, fontsize=35)
ax.tick_params(axis='both', which='major', labelsize=30)
ax2.tick_params(axis='both', which='major', labelsize=30)
ax.yaxis.label.set_color(p1.get_color())
ax2.yaxis.label.set_color(p2.get_color())
plt.grid(False)

plt.show()
plt.savefig('fig/relative_error.pdf', dpi=300)


##%
from utils.detection import return_cov_estimator
from matplotlib import transforms
from matplotlib.patches import Ellipse
def confidence_ellipse(ax, x=None, y=None, precomputed_stats=None, n_std=3.0, facecolor='none', **kwargs):
  """
  Create a plot of the covariance confidence ellipse of *x* and *y*.

  Parameters
  ----------
  x, y : array-like, shape (n, )
      Input data.

  ax : matplotlib.axes.Axes
      The axes object to draw the ellipse into.

  n_std : float
      The number of standard deviations to determine the ellipse's radiuses.

  **kwargs
      Forwarded to `~matplotlib.patches.Ellipse`

  Returns
  -------
  matplotlib.patches.Ellipse
  """
  if x.size != y.size:
    raise ValueError("x and y must be the same size")

  if precomputed_stats is None:
    cov = np.cov(x, y)
  else :
    cov = precomputed_stats['cov']
  pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
  # Using a special case to obtain the eigenvalues of this
  # two-dimensionl dataset.
  ell_radius_x = np.sqrt(1 + pearson)
  ell_radius_y = np.sqrt(1 - pearson)
  ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                    facecolor=facecolor, **kwargs)

  # Calculating the stdandard deviation of x from
  # the squareroot of the variance and multiplying
  # with the given number of standard deviations.
  scale_x = np.sqrt(cov[0, 0]) * n_std

  # calculating the stdandard deviation of y ...
  scale_y = np.sqrt(cov[1, 1]) * n_std
  if precomputed_stats is None:
    mean_x = np.mean(x)
    mean_y = np.mean(y)
  else:
    mean_x = precomputed_stats['mu'][0]
    mean_y = precomputed_stats['mu'][1]

  transf = transforms.Affine2D() \
    .rotate_deg(45) \
    .scale(scale_x, scale_y) \
    .translate(mean_x, mean_y)

  ellipse.set_transform(transf + ax.transData)
  return ax.add_patch(ellipse)

feats = np.loadtxt('fig/data/imdb-bert-feat.txt')
sample_idx = np.random.choice(range(len(feats)), size=200, replace=False)
sampled = feats[sample_idx, :-1]
labels = feats[sample_idx, -1]
all_labels = feats[:,-1]

MCD_kwargs = {'edgecolor': 'tab:red', 'linewidth': 1}
fig, ax = plt.subplots()

for cls_idx in np.unique(all_labels):
  cov_estim = return_cov_estimator('MCD')
  cov = cov_estim.fit(feats[all_labels==cls_idx, :-1]).covariance_
  loc = cov_estim.location_
  stats = {'cov': cov, 'mu':loc}

  for n_std in range(1, 4):
    confidence_ellipse(ax, feats[all_labels == cls_idx, 0], feats[all_labels == cls_idx, 1],n_std=n_std, linestyle='--', edgecolor='tab:blue', label='MLE')
    confidence_ellipse(ax, feats[all_labels == cls_idx, 0], feats[all_labels == cls_idx, 1], precomputed_stats=stats, n_std=n_std, label='MCD', **MCD_kwargs)


color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)
colors = np.random.rand(len(labels),4)
colors[labels==0] = color1
colors[labels==1] = color2
scatter = ax.scatter(sampled[:,0], sampled[:,1], c=colors)
handles, handle_labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], handle_labels[:2], title='Contour', title_fontsize=25, prop={'size':25})
ax.set_title('MLE/MCD Estimation (BERT on IMDB)', fontsize=30)
ax.set_xlim(-0.75,0.9)
ax.set_ylim(-0.3,0.4)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.tick_params(axis='both', direction='in', grid_alpha=0)
plt.savefig('fig/MCD.pdf', dpi=300)
