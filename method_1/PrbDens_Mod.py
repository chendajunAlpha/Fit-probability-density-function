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
    def __init__(self, read_sqlite_obj,
                 PrbDens_obj,
                 device,
                 learning_rate,
                 data_table_name,
                 uni_table_name,
                 part_size,
                 batch_size,
                 uniset_numperpart,
                 isdebug):

        self.__isdebug = isdebug
        self.__part_size = int(part_size)
        self.__batch_size = int(batch_size)
        self.__uniset_numperpart = int(uniset_numperpart)

        self.__read_train_data = read_sqlite_obj(data_table_name)
        self.__read_uni_data = read_sqlite_obj(uni_table_name)

        self.__PrbDens_obj = PrbDens_obj
        self.__device = device
        self.__criterion = nn.MSELoss()
        self.__opt = torch.optim.Adam(self.__PrbDens_obj.parameters(), lr=learning_rate)

    def __get_part_data(self,part_i):
        data_part = self.__read_train_data[part_i*self.__part_size:(part_i*self.__part_size+self.__part_size)]
        uni_part = []
        for i in range(self.__uniset_numperpart):
            uni_part = uni_part + self.__read_uni_data
        train_part = data_part + uni_part
        random.shuffle(train_part)
        label_list = []
        data_list = []
        for labeldata in train_part:
            label_tmp = labeldata[1]
            data_tmp = labeldata[2]
            label_list.append(label_tmp)
            data_list.append(data_tmp)
        self.__labels = np.array(label_list).astype(np.float32)
        self.__data = np.array(data_list).astype(np.float32)
        return len(train_part)


    def __call__(self):
        loss_batch = []
        random.shuffle(self.__read_train_data)
        for part_i in range(len(self.__read_train_data)//self.__part_size):
            len_train_part = self.__get_part_data(part_i)
            loss_batch = []
            for start in range(0, len_train_part, self.__batch_size):
                if (start + self.__batch_size) < len_train_part:
                    end = start + self.__batch_size
                else:
                    end = len_train_part
                label_tensor = torch.from_numpy(self.__labels[start:end].reshape(-1, 1)).to(self.__device)
                input_tensor = torch.from_numpy(self.__data[start:end].reshape(-1, 1)).to(self.__device)
                out_tensor = self.__PrbDens_obj(input_tensor)
                loss_tensor = self.__criterion(out_tensor, label_tensor)
                if self.__isdebug:
                    print("len_train_part = ", len_train_part, ",   part_i = ", part_i, ",   batch_start = ",start)
                    print("input = ", input_tensor,",   label = ", label_tensor,",  out_tensor = ",out_tensor)
                else:
                    loss_tensor.backward()
                    self.__opt.step()
                    self.__opt.zero_grad()
                loss_cpu_tensor = loss_tensor.to("cpu")
                loss_batch.append(loss_cpu_tensor.data.numpy())
        return loss_batch


class save_model_periodically(sql.sqlite_base):
    def __init__(self, sqlite_fullname,
                 save_model_fullname,
                 distrnn_obj,
                 hours_to_savemodel):
        super().__init__(sqlite_fullname)
        self.__hours_to_savemodel = hours_to_savemodel
        self.__save_model_fullname = save_model_fullname
        self.__distr_gen_obj = distrnn_obj
        self.__epoch_i_table_name = "epoch_i"
        self.__previous_time = time.localtime(time.time())

    def __call__(self, epoch_i):
        current_time = time.localtime(time.time())
        if not 0 <= (current_time.tm_min - self.__previous_time.tm_min) <= self.__hours_to_savemodel:
            # 0 <= (current_time.tm_sec - self.__previous_time.tm_sec) < self.__hours_to_savemodel:
            # 0 <= (current_time.tm_hour - self.__previous_time.tm_hour) <= self.__hours_to_savemodel:
            # 0 <= (current_time.tm_min - self.__previous_time.tm_min) <= self.__hours_to_savemodel:
            self.__previous_time = current_time
            self.save_model(epoch_i)

    def save_model(self,epoch_i):
        print("********** saving start ************")
        time_show = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(time_show, "   *************** saving epoch_i, epoch_i =", epoch_i)
        self.__create_epoch_i_sqlite(epoch_i)
        time_show = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(time_show, "   *************** saving to ", self.__save_model_fullname)
        torch.save(self.__distr_gen_obj.state_dict(), self.__save_model_fullname)
        print("********** saving end **************")

    def __create_epoch_i_sqlite(self, epoch_i):
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


