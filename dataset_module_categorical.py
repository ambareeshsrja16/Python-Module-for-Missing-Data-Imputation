import torch
import torch.utils.data as td
import numpy as np
import pandas as pd


class DataSetForImputation(td.Dataset):
    """
    Steps:
    1. Feed dataframe to class - the data frame should be the one with missing data (NaNs)
    2. Fill NaNs with appropriate placeholders (init)
    3. Implement get item with idx

    TO DO:
    :- Categorical variables - Replace categorical variables with mode, Others with mean
    :- Data Normalization between 0 and 1- DONE
    """

    def __init__(self, dataframe, one_hot_encoded_indexes): #normalize=False):
        super().__init__()
        self.orig_dataset = dataframe
        self.one_hot_encoded_indexes = one_hot_encoded_indexes
        
        self.perc_of_nans = sum((len(dataframe) - dataframe.count())) / dataframe.size
#         self.normalize = normalize # Don't normalize if categorical data is involved
        
        # Replacing NaNs with most common value if Categorical, if not mean value
        self.dataframe = dataframe.apply(lambda x: x.fillna(x.mode().iloc[0]) if x.dtype =='object' else x.fillna(x.mean()), axis=0) 
        
#         self.max_df = self.dataframe.max() 
#         self.min_df = self.dataframe.min()
        
#         if np.dtype(object) in set(self.dataframe.dtypes):
#             self.normalize = False
        
#         if self.normalize:  #Normalize dataframe elements to [0,1] -> Dividing by range
#             self.dataframe = (self.dataframe- self.dataframe.min())/(self.dataframe.max()-self.dataframe.min())
        
        self.columns = self.dataframe.columns


    def __repr__(self):
        return f"Dataframe Size:{len(self)}, Perc of NaNs: {self.perc_of_nans * 100:.2f}"

    def __len__(self):
        return len(self.dataframe)

    def variables(self):
        return self.dataframe.columns

    def __getitem__(self, idx):
        datapoint = torch.tensor(self.dataframe.iloc[idx])
        # convert to tensor
        # check https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor
        return datapoint, datapoint  # label and datapoint are the same
    
    def get_denormalized_data(self, norm_x):
        """Return x after denormalizing it"""
        
#         if isinstance(norm_x, torch.Tensor):
#             if len(norm_x.shape)==1:
#                 norm_x = norm_x.unsqueeze(0)
#             norm_x = norm_x.detach().cpu().numpy()
        
#         norm_x = pd.DataFrame(norm_x, columns = self.columns)
#         if self.normalize:
#             x = (norm_x)*(self.max_df - self.min_df) + self.min_df
#         else:
#             x = norm_x
#         return x

if __name__ == "__main__":
    np.random.seed(18)
    test_df = pd.DataFrame(np.random.rand(4,5))
    test_df.iloc[0,0] = np.NaN
    test_df.iloc[3,0] = np.NaN
    trainset = DataSetForImputation(test_df)
    print(trainset)
    print(trainset[0])
    print(trainset[3])