def get_dataframe_from_csv(filename, header_row=0, sep='\s+'):
    """
    input filename (full path) and returns dataframe with data
    TO DO:
        -: As of now reading headerless files with header = None, what if the data has a header, how to deal with that
        -: Should the last coloumn name be replaced with "label"?
        -: Add functionality for space de-limited or comma de-limited files
        -: Improve logging, make it module specific logging
    """
    assert isinstance(filename, str), "Input complete filename as a string"
    import pandas as pd
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logging.info("Input filename has to be space separated data")
    data_orig = pd.read_csv(filename, sep=sep, header=header_row)
    return data_orig


def induce_missingness(dataframe, perc_variables_sampled=0.5, threshold=0.2, logger_level=20):
    """
    Note dataframe doesn't have label
    Steps:
        1. Append random uniform vector to the dataframe
        2. Decide thresold (default = 20%)
        3. Sample variables (default = 50%)
        4. In those variables (from 3), check the last column and if the value is less than threshold (2), set them to NaN

    """
    import pandas as pd
    import numpy as np
    assert isinstance(dataframe, pd.DataFrame)
    import pandas as pd
    import logging
    logger = logging.getLogger()
    logger.setLevel(logger_level)

    RANDOM_SEED = 19
    np.random.seed(RANDOM_SEED)  # Reproducibility

    observations_number, variables_number = dataframe.shape[0], dataframe.shape[1]
    sampled_dataframe = dataframe.iloc[:, :].sample(n=int(variables_number * perc_variables_sampled),
                                                    axis=1)  # sample perc_variables_sampled
    sampled_variables = list(sampled_dataframe.columns)
    sampled_dataframe["random"] = np.random.uniform(size=observations_number)

    new_df = dataframe.copy()
    new_df.loc[sampled_dataframe["random"] < threshold, sampled_variables] = np.NAN

    logging.debug(f"\n{new_df.head()}")
    logging.debug(f"\n{sampled_dataframe.head()}")
    logging.debug(f"\n{dataframe.head()}")

    logging.info(" Returning new dataframe with missingness(MCAR) induced")

    perc_of_nans = sum((len(new_df) - new_df.count())) / new_df.size
    logging.info(f" Percentage of NaNs in returned dataframe : {perc_of_nans * 100:.2f}")

    return new_df


def create_train_test_split(dataframe, test_perc=0.3, logger_level=20):
    """
    Steps:
    1. Induce missingness in the dataframe <use induce_missingness>
    2. Split the resultant dataframe into train, test sets
    3. Return both along with a third - test set without missingness
    TO DO:
        -: Figure out better way to extract elements via indexing
    """
    import pandas as pd
    import numpy as np
    assert isinstance(dataframe, pd.DataFrame)

    import logging
    logger = logging.getLogger()
    logger.setLevel(logger_level)

    from sklearn.model_selection import train_test_split
    RANDOM_SEED = 18
    np.random.seed(RANDOM_SEED)  # Reproducibility

    _, full_test_df = train_test_split(dataframe, test_size=test_perc, random_state=RANDOM_SEED)
    train_df, test_df = train_test_split(induce_missingness(dataframe=dataframe), test_size=test_perc,
                                         random_state=RANDOM_SEED)
    # Used the same random_state to split the data twice in the same way, so full_test_df will be the filled part of dataframe

    '''
    TO DO: 
        -: Figure out why the following indexing is not working
    full_test_df = dataframe[dataframe.index.isin(test_df.index)]
    '''

    logging.info(
        f" Returning train_df, test_df, full_test_df after splitting dataframe in {1 - test_perc}/{test_perc} split ")
    logging.info(" Note: full_test_df is the same as test_df but without NaNs")
    return train_df, test_df, full_test_df

def convert_probabilities_to_one_hot(input_tensor, input_tensor_one_hot_indices):
    """Given prediction, correct indices of one-hot-encoded variables =>
    Returns predictions by rounding up the max probabilities to 1
    Creates new copy, doesn't modify original"""
    import torch
    
    #WRITE ASSERTS
    input_t  = input_tensor
    for i,j in input_tensor_one_hot_indices:
        temp_ind = torch.argmax(input_t[i:j])
        input_t[i:j]= 0.0
        input_t[i+temp_ind] = 1.0
    return input_t

def plot_loss_curve(filename = "DAE_Arch_N_7_ImputeOnlyNaNs_WithDropout/artifacts/loss_curve", title = "DAE_Arch_N_7_ImputeOnlyNaNs_WithDropout", freq = 1):
    """
    filename: model_name+"/artifacts/loss_curve"
    Plot the Loss vs Epoch curve   
    freq = 10 : Plotting values for every 10th epoch
    """
    assert isinstance(filename, str)
    assert isinstance(title, str)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    with open(filename,"r") as f:
        data = f.readlines()
    
    x = np.zeros(len(data))
    y = np.zeros_like(x)
    
    for i, item in enumerate(data):  #item ="Epoch_number: 0 Loss: 16.3756"
        data_elem = item.split()
        x[i] = int(data_elem[1])
        y[i] = float(data_elem[3])
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    plt.plot(x[::freq],y[::freq])
    plt.show()


def plot_train_and_val_loss_curves(filename1 = "DAE_Arch_N_7_ImputeOnlyNaNs_WithDropout/artifacts/loss_curve",
                    filename2 = "DAE_Arch_N_7_ImputeOnlyNaNs_WithDropout/artifacts/val_loss_curve",
                    title = "DAE_Arch_N_7_ImputeOnlyNaNs_WithDropout", annotation= None, freq = 1):
    """
    filename: model_name+"/artifacts/loss_curve"
    Plot the Loss vs Epoch curve   
    freq = 10 : Plotting values for every 10th epoch
    """
    assert isinstance(filename1, str)
    assert isinstance(filename2, str)
    assert isinstance(title, str)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    with open(filename1,"r") as f:
        data = f.readlines()
    
    with open(filename2,"r") as f:
        data_val = f.readlines()
    
    x = np.zeros(len(data))
    y = np.zeros_like(x)
    
    x_val = np.zeros(len(data_val))
    y_val = np.zeros_like(x_val)
    
    for i, item in enumerate(data):  #item ="Epoch_number: 0 Loss: 16.3756"
        data_elem = item.split()
        x[i] = int(data_elem[1])
        y[i] = float(data_elem[3])
    
    for i, item in enumerate(data_val):  #item ="Epoch_number: 0 Loss: 16.3756"
        data_elem = item.split()
        x_val[i] = int(data_elem[1])
        y_val[i] = float(data_elem[3])
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    plt.plot(x[::freq],y[::freq], label= "Training Loss") #3rd arg for color '-b'
    plt.plot(x_val[::freq], y_val[::freq], label = "Validation Loss")
    plt.legend(loc='upper right')
    
    if annotation:
        assert isinstance(annotation, str)
        plt.annotate(annotation, xy=(0.3, 0.9), xycoords='axes fraction') #Dataset
        
    plt.show()
      
    
if __name__ == "__main__":
    filename = "data/shuttle/shuttle_trn"
    train_df = get_dataframe_from_csv(filename).iloc[:, :-1]  # remove label
    print(train_df.head())

    df1 = train_df[:]
    df2 = induce_missingness(df1, logger_level=20)
    print(df1.head())
    print(df2.head())

    # Test
    a, b, c = create_train_test_split(df1)
    print(a.head())
    print(b.head())
    print(c.head())
