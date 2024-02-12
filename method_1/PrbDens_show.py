import os
import numpy as np
import torch
import PrbDens_Mod as PDM_cdj
from matplotlib import pyplot as plt


def main():
    device = "cuda:0"
    data_name = "mix_"
    model_type = "PrbDens_"
    NNModel_path = "NNModel"
    databasename = "distr"
    max_data = 15
    min_data = -1
    step_show = 30

    PrbDens_obj = PDM_cdj.distr_PrbDens()
    PrbDens_obj.to(device)

    NNModel_fullname = os.path.join(NNModel_path, data_name + model_type + databasename + ".plk")
    print("NNModel_fullname = ",NNModel_fullname)
    PrbDens_obj.load_state_dict(torch.load(NNModel_fullname))

    input = []
    for i in range(min_data*step_show,max_data*step_show):
        input.append(i/step_show)

    input_tensor = torch.from_numpy(np.array(input).reshape(-1,1).astype(np.float32)).to(device)
    outtensor = PrbDens_obj(input_tensor)
    outtensor_cpu = outtensor.to("cpu")
    output = outtensor_cpu.data.numpy()

    plt.plot(input, output)
    plt.show()


if __name__=="__main__":
    main()


