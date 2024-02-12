import os
import numpy as np
import torch
import sqlite_operator as sql
import PrbDens_Mod as PDM_cdj
from matplotlib import pyplot as plt


def main():
    device = "cuda:0"
    sqlite_path = "sqlites"
    model_type = "PrbDens_"
    NNModel_path = "NNModel"
    databasename = "distr"
    max_show = 15
    min_show = -5
    step_show = 30

    sqlite_fullname = os.path.join(sqlite_path, databasename + "_sqlite.db")
    read_sqlite_obj = sql.read_sqlite(sqlite_fullname)
    min_unif, max_unif = read_sqlite_obj.read_minmax()

    PrbDens_obj = PDM_cdj.distr_PrbDens()
    PrbDens_obj.to(device)

    NNModel_fullname = os.path.join(NNModel_path, model_type + databasename + ".plk")

    print("NNModel_fullname_1 = ", NNModel_fullname)

    PrbDens_obj.load_state_dict(torch.load(NNModel_fullname))

    input = []
    one = []
    PD_unif = []
    for i in range(min_show*step_show,max_show*step_show):
        input.append(i/step_show)
        one.append(1.0)
        PD_unif.append(1/(max_unif - min_unif))

    input_tensor = torch.from_numpy(np.array(input).reshape(-1,1).astype(np.float32)).to(device)
    one_tensor = torch.from_numpy(np.array(one).reshape(-1,1).astype(np.float32)).to(device)
    PD_unif_tensor = torch.from_numpy(np.array(PD_unif).reshape(-1, 1).astype(np.float32)).to(device)

    out_tensor_1 = torch.div(one_tensor,PrbDens_obj(input_tensor))
    out_tensor_2 = torch.sub(out_tensor_1,one_tensor)
    out_tensor = torch.mul(PD_unif_tensor,out_tensor_2)


    out_tensor_cpu = out_tensor.to("cpu")
    output = out_tensor_cpu.data.numpy()

    plt.plot(input, output)
    plt.show()


if __name__=="__main__":
    main()


