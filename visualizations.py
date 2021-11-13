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
p = ax.scatter(robust_spectrum.iloc[:,1], robust_spectrum.iloc[:,0], label='kPCA', s=50, alpha=0.5)
ax.scatter(naive_spectrum.iloc[:,1], naive_spectrum.iloc[:,0], label='Raw', s=50, alpha=0.5)
ax.set_title('Covariance Matrix Eigenvalues', fontsize=28)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Max. E.V', fontsize=25)
ax.set_xlabel('Min. E.V', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=35)
ax.legend(loc='best', prop={'size':25})

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

feats = np.loadtxt('fig/data/sst2-bert-feats.txt')
sample_idx = np.random.choice(range(len(feats)), size=40, replace=False)
sampled = feats[sample_idx, :-1]
labels = feats[sample_idx, -1]
all_labels = feats[:,-1]

MCD_kwargs = {'edgecolor': 'tab:red', 'linewidth': 1}
fig, ax = plt.subplots()

for cls_idx in np.unique(all_labels):
  cov_estim = return_cov_estimator('MCD', params={'h':None})
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
ax.set_title('MLE/MCD Estimation (SST2-BERT)', fontsize=30)
ax.set_xlim(-0.75,0.9)
ax.set_ylim(-0.3,0.4)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.tick_params(axis='both', direction='in', grid_alpha=0)
plt.savefig('fig/MCD.pdf', dpi=300)


# fig, ax = plt.subplots()
# cls = 1
# for h_ in [0.3, 0.5, 0.7, 0.9, 1.0]:
#   cov_estim = return_cov_estimator('MCD', params={'h':h_})
#   cov = cov_estim.fit(feats[all_labels==cls, :-1]).covariance_
#   loc = cov_estim.location_
#   stats = {'cov': cov, 'mu':loc}
#   color = plt.cm.magma(h_)
#   MCD_kwargs = {'edgecolor': color, 'linewidth': 2}
#   confidence_ellipse(ax, feats[all_labels == cls, 0], feats[all_labels ==cls, 1], precomputed_stats=stats,
#                      n_std=3, label=f'{h_}', **MCD_kwargs)
#
# # confidence_ellipse(ax, feats[all_labels == cls_idx, 0], feats[all_labels == cls_idx, 1],n_std=n_std, linestyle='--', edgecolor='tab:blue', label='MLE')
#
# single_cls_feats = feats[all_labels==cls]
# sample_idx = np.random.choice(range(len(single_cls_feats)), size=50, replace=False)
# sampled = single_cls_feats[sample_idx, :-1]
# labels = single_cls_feats[sample_idx, -1]
# scatter = ax.scatter(sampled[:,0], sampled[:,1], color=color1)
# ax.legend(prop={'size':22}, loc='upper left')
#
# ax.set_xlim(-0.6, -0.3)
# ax.set_ylim(-0.5,0.5)
# ax.tick_params(axis='both', direction='in', grid_alpha=0)
# ax.set_title("MCD Contour at Various $h$")
# plt.savefig("fig/MCD_at_various_h.pdf", dpi=300)

##%
path = 'runs/imdb/discussion/MCD/textattack-bert-base-uncased-imdb/textfooler/MCD-mahal.csv'
data = pd.read_csv(path)
color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)

steps = {"MCD": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, "MLE"], "PCA": [50, 100, 150, 200, 250, 300, "None"]}
fig, ax = plt.subplots(1,2, figsize=(5,5))
ax[0].plot(steps['MCD'][2:], data['auc'][2:], linestyle="dashed", c=color1, label='AUC')
ax[0].plot(steps['MCD'][2:], data['tpr'][2:], c=color1, label='TPR')

path = 'runs/imdb/discussion/kPCA/textattack-bert-base-uncased-imdb/textfooler/MCD-mahal.csv'
data = pd.read_csv(path)
ax[1].plot(steps['PCA'], data['tpr'], c=color1)
ax[1].plot(steps['PCA'], data['auc'], linestyle="dashed", c=color1)

ax[0].set_xlabel("Support Fraction ($h$)", fontsize=22)
ax[1].set_xlabel("kPCA Dimension ($P$)", fontsize=22)

ax[0].legend(prop={'size':20})
ax[0].tick_params(axis='x', which='major', labelsize=20)
# ax[1].set_yticks([])
# ax.tick_params(axis='both', direction='in', grid_alpha=0)
plt.setp(ax[1].get_yticklabels(), visible=False)
ax[1].sharey(ax[0])
ax[0].set_ylim(0.95, 0.98)
plt.savefig('fig/hyperparam.pdf', dpi=300)


##%
paths = ['fig/data/fae-bert-imdb.csv', 'fig/data/fae-roberta-imdb.csv', 'fig/data/fae-bert-agnews.csv', 'fig/data/fae-roberta-agnews.csv',\
         'fig/data/fae-bert-sst2.csv', 'fig/data/fae-roberta-sst2.csv']

fig, axes = plt.subplots(1,len(paths), sharey='all')
for idx in range(len(paths)):
  data = pd.read_csv(paths[idx], index_col=0)

  if 'sst2' in paths[idx]:
    x_labels = ['TF', 'PWWS', 'BAE']
  else:
    x_labels = ['TF', 'PWWS', 'BAE', 'TF-adj']
  x = np.arange(1,len(x_labels)+1)
  w = 0.3

  axes[idx].bar(x-w, data.loc['original'], w, label='Original')
  axes[idx].bar(x, data.loc['RDE'], w, label='RDE')
  axes[idx].bar(x+w, data.loc['MLE'], w, label='MLE')
  axes[idx].set_xticks(range(1,len(x_labels)+1))
  axes[idx].set_ylim(0.5,1)

axes[0].set_ylabel('AUC')
fig.supxlabel('', fontsize=25, weight='bold')
plt.subplots_adjust(wspace=0.01, hspace=0.05)
fig.text(0.185, 0.93, 'IMDB', fontsize=20, weight='bold')
axes[0].set_title('BERT', fontsize=20)
axes[1].set_title('ROBERTa', fontsize=20)
fig.text(0.49, 0.93, 'AG-News', fontsize=20, weight='bold')
axes[2].set_title('BERT', fontsize=18)
axes[3].set_title('ROBERTa', fontsize=18)
fig.text(0.81, 0.93, 'SST-2', fontsize=20, weight='bold')
axes[4].set_title('BERT', fontsize=18)
axes[5].set_title('ROBERTa', fontsize=18)

axes[-1].legend(loc='lower center')
fig.savefig('fig/fae.pdf', dpi=300)


##
our_conf = np.load("fig/data/ours-conf.npy")
labels = np.load("fig/data/labels.npy")

conf = our_conf
fig, ax = plt.subplots(figsize=(5,5))
kwargs = dict(histtype='stepfilled', alpha=0.3, bins=30, density=False)

x2 = conf[labels==1]
ax.hist(x=x2, label='adv.', **kwargs)
x1 = conf[labels==0]
ax.hist(x=x1, label='clean', **kwargs)
ax.set_xticks([])
ax.set_xlabel(r'log$p_{\theta}(z)$')

ax.legend(prop={'size':30}, loc='upper left')

fig.savefig('fig/confidence.pdf', dpi=300)

###
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import gridspec
# Load our data
data_list = ['imdb', 'ag-news', 'sst2']
attack_list = ['tf', 'pwws']
fig = plt.figure()

ax_cnt = 1
for r_idx, d_name in enumerate(data_list):
  for c_idx, a_name in enumerate(attack_list):
    small_ax = fig.add_subplot(3,2,ax_cnt)
    small_ax.set_aspect('equal')
    ax_cnt +=1
    # d_name, a_name = 'imdb', 'tf'
    conf_path = f"fig/data/roc/{d_name}/{a_name}/conf.csv"
    gt_path = f"fig/data/roc/{d_name}/{a_name}/gt.csv"
    data = np.loadtxt(conf_path)
    gt = np.loadtxt(gt_path)
    fgws_path = f"fig/data/roc/{d_name}/{a_name}/scores-0.csv"
    fgws = np.loadtxt(fgws_path, delimiter=',')
    gts = [gt, gt, fgws[:,1]]
    scores = [data[:,1], data[:,0], -fgws[:,0]]
    names = ['RDE', r'$-$MCD', 'FGWS']

    small_ax.set_yticks(np.arange(0, 1.1, 0.1), minor=True)
    small_ax.set_xticks(np.arange(0, 1.1, 0.1), minor=True)
    small_ax.grid(which='minor', alpha=1.0)

    #Compute
    for gt, score, name in zip(gts, scores, names):
      fpr, tpr, threshold = roc_curve(gt, -score)
      auc = roc_auc_score(gt, -score)

      #plot parameters
      lw = 2

      name_plus_auc = f"{name}: {auc*100:.1f}"
      small_ax.plot(fpr, tpr, label=name_plus_auc)
      small_ax.plot([0, 1], [0, 1], color="grey", lw=lw, linestyle="--")


ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")

all_axes = fig.get_axes()
all_axes[0].set_ylabel("IMDB")
all_axes[2].set_ylabel("AG-News")
all_axes[4].set_ylabel("SST-2")

all_axes[0].set_xlabel("TF")
all_axes[0].xaxis.set_label_position('top')

all_axes[1].set_xlabel("PWWS")
all_axes[1].xaxis.set_label_position('top')

for s_ax in all_axes:
  s_ax.legend(prop={'size':13})

all_axes[1].set_yticks([])
all_axes[3].set_yticks([])
all_axes[5].set_yticks([])

fig.savefig('fig/ROC.pdf', dpi=300)
