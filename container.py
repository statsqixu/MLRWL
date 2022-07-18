"""
ITR Dataset
"""

# Author: Qi Xu <qxu6@uci.edu>



from torch.utils.data import Dataset


# Define the dataset
class ITRDataset(Dataset):
    
    def __init__(self, R, X, A, W):
        
        self.output, self.covariate, self.treatment, self.weight = R, X, A, W
        
    def __len__(self):
        return len(self.output)
    
    def __getitem__(self, idx):
        return self.output[idx], self.covariate[idx], self.treatment[idx], self.weight[idx]