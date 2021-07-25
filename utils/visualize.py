#PCA

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
pca = PCA(n_components=2)
tsne = TSNE(n_components=2)
pca.fit(test_features.numpy())
X_reduced = pca.fit_transform(test_features.numpy())
y_adv = np.array(testset.result_type.values==1, dtype=bool)
y_incorrect = np.array(testset.result_type.values==-1, dtype=bool)
y_correct = np.array(testset.result_type.values==0, dtype=bool)
y_gt = testset.ground_truth_output
plt.scatter(X_reduced[y_correct,0], X_reduced[y_correct,1], c=y_gt[y_correct])
plt.scatter(X_reduced[y_adv,0], X_reduced[y_adv,1], c=y_gt[y_adv], alpha=0.2, marker='*')
plt.scatter(X_reduced[y_incorrect,0], X_reduced[y_incorrect,1], c=y_gt[y_incorrect], alpha=1.0, marker='.')
plt.show()
plt.savefig("PCA_result.png")