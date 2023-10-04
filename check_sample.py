import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# features =  np.array([1, 2,3,4,5,6,7,8,9,10]).reshape(-1, 1)
# features =  np.array([[1,1], [2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1],[10,1]])
# features = [[torch.tensor([[1], [2]]), torch.tensor([2])], [torch.tensor([[1], [2]]), torch.tensor([2])], [torch.tensor([[1], [2]]), torch.tensor([2])], [torch.tensor([[1], [2]]), torch.tensor([2])], [torch.tensor([[1], [2]]), torch.tensor([2])],
#             [torch.tensor([[1], [2]]), torch.tensor([2])], [torch.tensor([[1], [2]]), torch.tensor([2])], [torch.tensor([[1], [2]]), torch.tensor([2])], [torch.tensor([[1], [2]]), torch.tensor([2])], [torch.tensor([[1], [2]]), torch.tensor([2])]]
# features = [[[[1], [2]], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]],
#             [[[1], [2]], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]]]
# features = {
#     'first': [torch.tensor([[1], [2]]),torch.tensor([[1], [2]]),torch.tensor([[1], [2]]),torch.tensor([[1], [2]]),torch.tensor([[1], [2]]),torch.tensor([[1], [2]]),torch.tensor([[1], [2]]),torch.tensor([[1], [2]]),torch.tensor([[1], [2]]),torch.tensor([[1], [2]])],
#     'second': [torch.tensor([2]),torch.tensor([2]),torch.tensor([2]),torch.tensor([2]),torch.tensor([2]),torch.tensor([2]),torch.tensor([2]),torch.tensor([2]),torch.tensor([2]),torch.tensor([2])]
# }
# features = {
#     'first': [2,2,2,2,2,2,2,2,2,2],
#     'second': [2,2,2,2,2,2,2,2,2,2]
# }
features=[{'first':[1,2], 'second':[1,2]},{'first':[1,2], 'second':[1,2]},{'first':[1,2], 'second':[1,2]},{'first':[1,2], 'second':[1,2]},{'first':[1,2], 'second':[1,2]},{'first':[1,2], 'second':[1,2]},{'first':[1,2], 'second':[1,2]},{'first':[1,2], 'second':[1,2]},{'first':[1,2], 'second':[1,2]},{'first':[1,2], 'second':[1,2]}]
# features = np.array([[1,2], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]],[[[1], [2]], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]], [[[1], [2]], [2]])
features = pd.DataFrame(features)
# features.reset_index()
labels = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
oversampler = SMOTE(random_state=22, k_neighbors=2)
os_features, os_labels = oversampler.fit_resample(features, labels)
os_features_train, os_features_val, os_labels_train, os_labels_val = train_test_split(os_features, os_labels,
                                                                                      test_size=0.2,
                                                                                      random_state=22)
print("1")
