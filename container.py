"""
ITR Dataset
"""

# Author: Qi Xu <qxu6@uci.edu>



from torch.utils.data import Dataset


# Define the dataset
class ITRDataset(Dataset):
    
    def __init__(self, Y, X, A):
        
        self.output, self.covariate, self.treatment = Y, X, A
        
    def __len__(self):
        return len(self.output)
    
    def __getitem__(self, idx):
        return self.output[idx], self.covariate[idx], self.treatment[idx]