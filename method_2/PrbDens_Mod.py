import time
import sqlite3
import numpy as np
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as fun
import sqlite_operator as sql


class distr_PrbDens(nn.Module):
    def __init__(self):
        super().__init__()
        self.hiden1 = nn.Linear(1, 128)
        self.hiden2 = nn.Linear(128, 256)
        self.hiden3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)

    def forward(self,input_tensor):
        out = fun.relu(self.hiden1(input_tensor))
        out = fun.relu(self.hiden2(out))
        out = fun.relu(self.hiden3(out))
        return fun.relu(self.output(out))


class train_distr_PrbDens:
    def __init__(self,
                 device,
                 learning_rate,
                 batch_size,
                 read_sqlite_obj,
                 main_table_name,
                 unif_table_name,
                 isdebug):

        self.device = device
        self.learning_rate=learning_rate
        self.batch_size = int(batch_size)
        self.read_sqlite_obj = read_sqlite_obj
        self.train_data = self.read_sqlite_obj(main_table_name)
        self.train_data = self.train_data + self.read_sqlite_obj(unif_table_name)
        self.isdebug = isdebug


    def train(self):
        len_train_data = len(self.train_data)
        loss_batch = []
        for start in range(0, len_train_data, self.batch_size):
            if (start + self.batch_size) < len_train_data:
                end = start + self.batch_size
            else:
                end = len_train_data
            label_tensor = torch.from_numpy(self.__labels[start:end].reshape(-1, 1)).to(self.device)
            input_tensor = torch.from_numpy(self.__data[start:end].reshape(-1, 1)).to(self.device)
            out_tensor = self.__PrbDens_obj(input_tensor)
            loss_tensor = self.__criterion(out_tensor, label_tensor)
            if self.isdebug:
                print("len_train_data = ", len_train_data, "    batch_start = ",start)
                print("input = ", input_tensor,"    label = ", label_tensor)
            else:
                loss_tensor.backward()
                self.__opt.step()
                self.__opt.zero_grad()
            loss_cpu_tensor = loss_tensor.to("cpu")
            loss_batch.append(loss_cpu_tensor.data.numpy())
        return loss_batch

    def __get_labels_data(self):
        random.shuffle(self.train_data)
        label_list = []
        data_list = []
        for labeldata in self.train_data:
            label_tmp = labeldata[1]
            data_tmp = labeldata[2]
            label_list.append(label_tmp)
            data_list.append(data_tmp)
        self.__labels = np.array(label_list).astype(np.float32)
        self.__data = np.array(data_list).astype(np.float32)

    def __call__(self,PrbDens_obj):
        self.__criterion = nn.MSELoss()
        self.__PrbDens_obj = PrbDens_obj
        self.__opt = torch.optim.Adam(PrbDens_obj.parameters(), lr=self.learning_rate)
        self.__get_labels_data()


class save_model_periodically(sql.sqlite_base):
    def __init__(self,
                 sqlite_fullname,
                 hours_to_savemodel):
        super().__init__(sqlite_fullname)
        self.__hours_to_savemodel = hours_to_savemodel
        self.__epoch_i_table_name = "epoch_i"
        self.__previous_time = time.localtime(time.time())

    def __call__(self,save_model_fullname, PrbDens_obj,epoch_i):
        current_time = time.localtime(time.time())
        if not 0 <= (current_time.tm_min - self.__previous_time.tm_min) <= self.__hours_to_savemodel:
            # 0 <= (current_time.tm_sec - self.__previous_time.tm_sec) < self.__hours_to_savemodel:
            # 0 <= (current_time.tm_hour - self.__previous_time.tm_hour) <= self.__hours_to_savemodel:
            # 0 <= (current_time.tm_min - self.__previous_time.tm_min) <= self.__hours_to_savemodel:
            self.__previous_time = current_time
            self.save_epoch_i(epoch_i)
            self.save_model(save_model_fullname, PrbDens_obj)

    def save_model(self,save_model_fullname,PrbDens_obj):
        time_show = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(time_show, "   *************** saving to ", save_model_fullname)
        torch.save(PrbDens_obj.state_dict(), save_model_fullname)
        print("********** saving end **************")

    def save_epoch_i(self, epoch_i):
        print("********** saving start ************")
        time_show = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(time_show, "   *************** saving epoch_i, epoch_i =", epoch_i)
        sqlite_obj = sqlite3.connect(self.sqlite_fullname)
        delete_table_sqlcom = "DROP TABLE IF EXISTS %s" % self.__epoch_i_table_name
        sqlite_obj.execute(delete_table_sqlcom)
        create_epochi_sqlcom = "CREATE TABLE %s ( epoch INTEGER )" % self.__epoch_i_table_name
        sqlite_obj.execute(create_epochi_sqlcom)
        self.insert_into_sqlite(sqlite_obj,self.__epoch_i_table_name,["epoch"],[epoch_i])
        sqlite_obj.commit()
        sqlite_obj.close()

    def read_epoch_i(self):
        sqlite_obj = sqlite3.connect(self.sqlite_fullname)
        create_epochi_sqlcom = "CREATE TABLE IF NOT EXISTS %s ( epoch_i INTEGER )" % self.__epoch_i_table_name
        sqlite_obj.execute(create_epochi_sqlcom)
        read_epoch_i = self.read_from_sqlite(sqlite_obj, self.__epoch_i_table_name)
        sqlite_obj.commit()
        sqlite_obj.close()
        return read_epoch_i


