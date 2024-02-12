import os
import numpy as np
import torch
import PrbDens_Mod as PDM_cdj
import sqlite_operator as sql


isdebug = False
sqlite_path = "sqlites"
NNModel_path = "NNModel"
main_table_name = "main_distr"
unif_table_name = "unif_distr"

device = "cuda:0"
learning_rate = 0.0001
hours_to_savemodel = 5

if isdebug:
    model_type = ""
    databasename = "debug"
    batch_size = 10
    epoches = 2
else:
    model_type = "PrbDens_"
    databasename = "distr"
    batch_size = 3000
    epoches = 1000


def main():
    # Read the sample sets from sqlite.db
    sqlite_fullname = os.path.join(sqlite_path, databasename + "_sqlite.db")
    read_sqlite_obj = sql.read_sqlite(sqlite_fullname)
    # Build a Neural Networks
    PrbDens_obj= PDM_cdj.distr_PrbDens()
    PrbDens_obj.to(device)

    # When the Neural Network state path doesn't exist ,create it
    if os.path.exists(NNModel_path) is not True:
        os.mkdir(NNModel_path)

    # When the model state files exist ,load them
    save_model_fullname = os.path.join(NNModel_path, model_type + databasename +  ".plk")
    if os.path.exists(save_model_fullname):
        print("load the model_state")
        PrbDens_obj.load_state_dict(torch.load(save_model_fullname))

    # Build an object to save the models periodically
    save_obj = PDM_cdj.save_model_periodically(sqlite_fullname,hours_to_savemodel)
    epoch_i_tmp = save_obj.read_epoch_i()
    if len(epoch_i_tmp) == 0:
        epoch_i = 0
    else:
        epoch_i = epoch_i_tmp[0][0]
        if epoch_i==epoches:
            epoch_i = 0

    # Build a training object
    train_PrbDens_obj = PDM_cdj.train_distr_PrbDens(
                                                    device,
                                                    learning_rate,
                                                    batch_size,
                                                    read_sqlite_obj,
                                                    main_table_name,
                                                    unif_table_name,
                                                    isdebug)

    train_PrbDens_obj(PrbDens_obj)
    # Training
    loss_final = []
    for epoch in range(epoch_i, epoches):
        print("epoch = ", epoch)
        loss_batch = train_PrbDens_obj.train()
        print("loss of this epoch = ", np.mean(loss_batch))

        save_obj(save_model_fullname, PrbDens_obj, epoch)
        loss_final.append(np.mean(loss_batch))

    save_obj.save_epoch_i(epoch_i)
    save_obj.save_model(save_model_fullname, PrbDens_obj)
    print("***** final loss = ",np.mean(loss_final))

if __name__ == '__main__':
    main()





