import os
import numpy as np
import torch
import PrbDens_Mod as PDM_cdj
import sqlite_operator as sql


isdebug = False
data_name = "mix_"
sqlite_path = "sqlites"
NNModel_path = "NNModel"
uni_table_name = "uni_distr"
data_table_name = data_name + "distr"

device = "cuda:0"
learning_rate = 0.000001
hours_to_savemodel = 10
max_data = 15
min_data = -10

if isdebug:
    model_type = ""
    databasename = "debug"
    batch_size = 10
    epoches = 2
    big_num = 3
    sample_num_expected = 30  # sample_num_expected >= (uni_part_size_expected//big_num)
    uniset_size_expected = 23  # uniset_size_expected >= big_num
    uni_part_size_expected = 52  # (sample_num_expected*big_num) >= uni_part_size_expected >= uniset_size_expected
else:
    model_type = "PrbDens_"
    databasename = "distr"
    batch_size = 3000
    epoches = 1000
    big_num = 1000 # big_num is the n2/n1
    # Make sure the following sizes fit the memory size of my computer.
    sample_num_expected = 100000  # sample_num_expected >= (uni_part_size_expected//big_num)
    uniset_size_expected = 1000000  # uniset_size_expected >= big_num
    uni_part_size_expected = 10000000  # (sample_num_expected*big_num) >= uni_part_size_expected >= uniset_size_expected
uniset_size_base = uniset_size_expected//big_num
uniset_size = uniset_size_base*big_num
uni_part_size_base = uni_part_size_expected//uniset_size
uni_part_size = uni_part_size_base*uniset_size
part_size = uni_part_size/big_num
uniset_numperpart = uni_part_size/uniset_size
sample_num_base = sample_num_expected//part_size
sample_num = sample_num_base*part_size


def main():
    print("sample_num = ", sample_num, ",  uniset_size = ", uniset_size, ",   uni_part_size = ", uni_part_size)
    print("part_size = ", part_size, ",    uniset_numperpart = ", uniset_numperpart)

    # Read the Gaussian Mixture set ,min value of the Gaussian Mixture set ,
    # max value of the Gaussian Mixture set, and the uniform distribution set
    sqlite_fullname = os.path.join(sqlite_path, databasename + "_sqlite.db")
    read_sqlite_obj = sql.read_sqlite(sqlite_fullname)
    # Build a Neural Network to produce the Probability Density of the input of distribution
    PrbDens_obj = PDM_cdj.distr_PrbDens()
    PrbDens_obj.to(device)

    # When the Neural Network state path doesn't exist ,create it
    if os.path.exists(NNModel_path) is not True:
        os.mkdir(NNModel_path)
    # When the model state file exists ,load it
    NNModel_fullname = os.path.join(NNModel_path, data_name+ model_type + databasename +  ".plk")
    if os.path.exists(NNModel_fullname):
        print("load the model_state")
        PrbDens_obj.load_state_dict(torch.load(NNModel_fullname))

    # Build an object for saving the model periodically
    save_obj = PDM_cdj.save_model_periodically(sqlite_fullname,
                                               NNModel_fullname,
                                               PrbDens_obj,
                                               hours_to_savemodel)
    epoch_i_tmp = save_obj.read_epoch_i()
    if len(epoch_i_tmp) == 0:
        epoch_i = 0
    else:
        epoch_i = epoch_i_tmp[0][0]
        if epoch_i==epoches:
            epoch_i = 0

    # Build a training object
    train_PrbDens_obj = PDM_cdj.train_distr_PrbDens(read_sqlite_obj,
                                                    PrbDens_obj,
                                                    device,
                                                    learning_rate,
                                                    data_table_name,
                                                    uni_table_name,
                                                    part_size,
                                                    batch_size,
                                                    uniset_numperpart,
                                                    isdebug)

    # Training
    loss_final = []
    for epoch in range(epoch_i, epoches):
        print("epoch = ", epoch)
        loss_batch = train_PrbDens_obj()
        print("loss of this epoch = ", np.mean(loss_batch))
        save_obj(epoch)
        loss_final.append(np.mean(loss_batch))
    save_obj.save_model(epoches)
    print("***** final loss = ",np.mean(loss_final))


if __name__ == '__main__':
    main()





