import pandas as pd
import torch
import utils, dataset_module
import Modelling


def test_model(net=None, test_df=None, trainset=None, device=None, logger_level=20):
    """
    Function to evaluate the model on test set
    Should meet the following requirements:
        :-trainset - should be an object of class DataSetForImputation
        :-test_df - should be a Pandas dataframe with NaNs (if there are no NaNs, the same will be returned)
        :-net - should be an object of DenoisingAutoEncoder
    """
    assert isinstance(net, Modelling.DenoisingAutoEncoder)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(trainset, dataset_module.DataSetForImputation)

    import logging
    logger = logging.getLogger()
    logger.setLevel(logger_level)

    NaN_test_df = test_df.reset_index(drop=True)  # Dropping index so that everything is reindexed from 0

    test_df = test_df.apply(lambda x: x.fillna(x.mean()), axis=0)
    test_df_norm = (test_df - trainset.min_df) / (trainset.max_df - trainset.min_df)
    test_df_tensor = torch.tensor(test_df_norm.values).to(device)
    
    net = net.to(device)
    net = net.eval()
    logging.debug(f"{test_df_tensor.shape}")
    pred = net(test_df_tensor)

    pred = trainset.get_denormalized_data(pred)  # Predicted dataframe from the mode

    # Replace the NaNs in the original test_df with newly imputed values
    final_pred = NaN_test_df.where(~NaN_test_df.isna(), other=pred)
    logging.debug(f"final_pred:\n {final_pred.head()}")

    return final_pred


'''
TO DO:
:- Nesterov Momentum + Adam- Pytorch? Decay factor?
'''
import os
from tqdm import tqdm_notebook as tqdm


def train_model(start_steps=0, end_steps=5, net=None, model_name="DAE_Arch_N_7_ImputeOnlyNaNs_WithDropout",
                train_loader= None, val_loader=None, logger_level=20):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logger_level)

    NaN_flag = False

    for epoch in tqdm(range(start_steps, end_steps)):
        count = epoch - start_steps + 1
        net.train()
        # Epoch begins
        epoch_loss = 0.0
        for x, d in tqdm(train_loader):
            # Normalize between [0,1] for better convergence
            original_x = x
            x[torch.isnan(x)] = 0  # If an entire column is zero, division by 0, replace NaNs with zero
            d[torch.isnan(d)] = 0

            optimizer.zero_grad()
            x = x.to(device)
            with torch.no_grad():
                d = d.to(device)
            y = net(x)
            loss = torch.sqrt(criterion(y, d))  # RMSE Loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Break if NaN encountered
            if torch.isnan(loss) or torch.isinf(loss):
                logging.info(f"Loss value: {loss.item()}")
                logging.info("NaN/inf occured at:")
                logging.info(f"{x}\n")
                logging.info(f"{d}\n")
                logging.info(f"Original x was : {original_x}")
                NaN_flag = True
                break

            logging.debug(f"Count: {count}, Loss :{loss}")

        if NaN_flag: break  # Stop training if NaN encountered

        # Print to screen every few epochs
        if count % LOG_INTERVAL == 0:
            print(f"Epoch number:{epoch} Loss: {epoch_loss:.4f}")

            # Training artifacts
        if model_name not in os.listdir():
            os.makedirs(model_name + "/artifacts/saved_model/")

        # Write to loss file every epoch
        with open(model_name + "/artifacts/loss_curve", mode='a+') as f:
            f.write(f"Epoch_number: {epoch} Loss: {epoch_loss:.4f}\n")

        # Validation curve
        val_loss = 0.0
        net.eval()
        for x, d in val_loader:
            x[torch.isnan(x)] = 0
            d[torch.isnan(d)] = 0
            x = x.to(device)
            with torch.no_grad():
                d = d.to(device)
            y = net(x)
            loss = torch.sqrt(criterion(y, d))
            val_loss += loss
        net.train()
        # Write Val loss to file every epoch
        with open(model_name + "/artifacts/val_loss_curve", mode='a+') as f:
            f.write(f"Epoch_number: {epoch} Loss: {val_loss:.4f}\n")

        # Save model every few epochs
        if epoch % SAVE_INTERVAL == 0:
            torch.save(net.state_dict(), f"./{model_name}/artifacts/saved_model/model_at_epoch{epoch}")
        # Epoch Ends


if __name__ == "__main__":
    import torch.utils.data as td
    import numpy as np
    from torch.optim import Adam

    # Settings for device, randomization seed, default tensor type, kwargs for memory #DevSeedTensKwargs
    RANDOM_SEED = 18
    np.random.seed(RANDOM_SEED)

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        device = 'cpu'
        torch.manual_seed(RANDOM_SEED)
        torch.set_default_tensor_type(torch.FloatTensor)
        kwards = {}

    trainset = dataset_module.DataSetForImputation(train_df, normalize=True)
    testset = dataset_module.DataSetForImputation(test_df, normalize=True)

    LR = 1e-3
    DATAPOINTS = len(trainset)
    BATCH_SIZE = 512
    BATCHES = DATAPOINTS / BATCH_SIZE
    VARIABLES = len(trainset.variables())  # 9

    import Modelling

    net = Modelling.DenoisingAutoEncoder(len(trainset.variables()), theta=7, input_dropout=0.5)

    criterion = nn.MSELoss()
    net = net.to(device)


    train_loader = td.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val_loader = td.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    optimizer = Adam(net.parameters(), lr=LR)

    LOG_INTERVAL = 10
    SAVE_INTERVAL = 50
    torch.set_printoptions(sci_mode=False)