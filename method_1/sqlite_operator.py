import os
import sqlite3
import numpy as np
from numpy import random


class sqlite_base:
    def __init__(self,sqlite_fullname):
        self.sqlite_fullname = sqlite_fullname

    def insert_into_sqlite(self, sqlite_obj, table_name, column_list, row_list):
        len_column_list = len(column_list)
        column_names = column_list[0]
        values_tmp = " ?"
        del column_list[0]
        for column in column_list:
            column_names = column_names + " , " + column
            values_tmp = values_tmp + " , " + "?"
        cursor_obj = sqlite_obj.cursor()
        if len_column_list>1:
            insert_table_sql = "INSERT INTO " + table_name + " ( " + column_names + " )" + " VALUES (" + values_tmp + ");"  # (?, ?);
            cursor_obj.executemany(insert_table_sql, row_list)
        else:
            insert_table_sql = "INSERT INTO " + table_name + " ( " + column_names + " )" + " VALUES (" + str(row_list[0]) + ");"
            cursor_obj.execute(insert_table_sql)
        cursor_obj.close()

    def read_from_sqlite(self,sqlite_obj, table_name, result_column='*', condition_column=None,condition_value=None):
        cursor_obj = sqlite_obj.cursor()
        if not condition_column == None:
            if condition_value == None:
                raise ValueError("condition_column != None but condition_value == None")
            else:
                if isinstance(condition_value, str):
                    condition_value = "'" + condition_value + "'"
                else:
                    condition_value = str(condition_value)

                condition_str = " where " + condition_column + " = " + condition_value
        else:
            if not condition_value == None:
                raise ValueError("condition_column == None but condition_value != None")
            else:
                condition_str = ""
        select_str = "select " + result_column + " from " + table_name + condition_str
        cursor_obj.execute(select_str)
        fetchall_cdj = cursor_obj.fetchall()
        cursor_obj.close()
        sqlite_obj.commit()
        return fetchall_cdj


class create_sqlite_distr(sqlite_base):
    def __init__(self,
                 sqlite_fullname,
                 mix_table_name,
                 uni_table_name,
                 min_data,
                 max_data,
                 sample_num,
                 uniset_size,
                 big_num,
                 scale,
                 isdebug):
        super().__init__(sqlite_fullname)
        self.__minmax_table_name = "minmax_data"
        self.mix_table_name = mix_table_name
        self.uni_table_name = uni_table_name
        self.__min_data = min_data
        self.__max_data = max_data
        self.__scale = scale
        self.__isdebug = isdebug
        self.__sample_num = int(sample_num)
        self.__uniset_size = int(uniset_size)
        self.__big_num = int(big_num)

        self.__create_dataset()
        self.__create_data_table(self.mix_table_name)
        self.__create_data_table(self.uni_table_name)
        self.__create_minmax_table()

    def __create_dataset(self):
        self.__disrt_mix = []
        self.__distr_uni = []
        if self.__isdebug:
            self.__disrt_mix = list(range(self.__sample_num))
            self.__distr_uni = np.ones(self.__uniset_size)*100
        else:
            for i in range(self.__sample_num):
                randomint = random.randint(1, 11)
                match randomint:
                    case 1:
                        disrt_mix_tmp = random.normal(loc=3, scale=self.__scale)
                        while not self.__min_data < disrt_mix_tmp < self.__max_data:
                            disrt_mix_tmp = random.normal(loc=3, scale=self.__scale)
                    case 2|3:
                        disrt_mix_tmp = random.normal(loc=1, scale=self.__scale)
                        while not self.__min_data < disrt_mix_tmp < self.__max_data:
                            disrt_mix_tmp = random.normal(loc=1, scale=self.__scale)
                    case 4|5|6:
                        disrt_mix_tmp = random.normal(loc=6, scale=self.__scale)
                        while not self.__min_data < disrt_mix_tmp < self.__max_data:
                            disrt_mix_tmp = random.normal(loc=6, scale=self.__scale)
                    case 7|8|9|10:
                        disrt_mix_tmp = random.normal(loc=9, scale=self.__scale)
                        while not self.__min_data < disrt_mix_tmp < self.__max_data:
                            disrt_mix_tmp = random.normal(loc=9, scale=self.__scale)
                    case _ :
                        raise ValueError("randomint is wrong")
                self.__disrt_mix.append(disrt_mix_tmp)
            self.__distr_uni = [i*(self.__max_data - self.__min_data+2)/self.__uniset_size +
                                self.__min_data -1 for i in range(self.__uniset_size)]

    def __create_data_table(self,data_table_name):
        match data_table_name:
            case self.uni_table_name:
                samples = self.__distr_uni
                label = 0.0
            case self.mix_table_name:
                samples = self.__disrt_mix
                if self.__isdebug:
                    label = 1.0
                else:
                    label = self.__big_num/(self.__max_data-self.__min_data)
            case _ :
                raise ValueError("data_table_name is wrong")

        sqlite_obj = sqlite3.connect(self.sqlite_fullname)
        delete_table_sqlitecom = "DROP TABLE IF EXISTS %s" % data_table_name
        sqlite_obj.execute(delete_table_sqlitecom)
        create_table_sqlcom = "CREATE TABLE IF NOT EXISTS %s ( read_id INTEGER PRIMARY KEY AUTOINCREMENT, label REAL, sample REAL )" % data_table_name
        sqlite_obj.execute(create_table_sqlcom)

        column_list = ['label','sample']
        row_list = []
        for sample in samples:
            row_list.append((label,sample))
        self.insert_into_sqlite(sqlite_obj, table_name=data_table_name, column_list=column_list,row_list=row_list)

        sqlite_obj.commit()
        sqlite_obj.close()

    def __create_minmax_table(self):
        sqlite_obj = sqlite3.connect(self.sqlite_fullname)
        delete_table_com = "DROP TABLE IF EXISTS %s" % self.__minmax_table_name
        sqlite_obj.execute(delete_table_com)
        delete_table_com = "CREATE TABLE %s ( min_data REAL,max_data REAL )" % self.__minmax_table_name
        sqlite_obj.execute(delete_table_com)

        column_list = ["min_data","max_data"]
        row_list = [(self.__min_data, self.__max_data)]
        self.insert_into_sqlite(sqlite_obj, table_name=self.__minmax_table_name, column_list=column_list,row_list=row_list)

        sqlite_obj.commit()
        sqlite_obj.close()


class read_sqlite(sqlite_base):
    def __init__(self,sqlite_fullname):
        super().__init__(sqlite_fullname)
        self.__minmax_table_name = "minmax_data"
        sqlite_obj = sqlite3.connect(self.sqlite_fullname)
        minmax_data_tmp = self.read_from_sqlite(sqlite_obj, table_name=self.__minmax_table_name)
        sqlite_obj.commit()
        sqlite_obj.close()
        self.min_data = minmax_data_tmp[0][0]
        self.max_data = minmax_data_tmp[0][1]

    def __call__(self,data_table_name):
        sqlite_obj = sqlite3.connect(self.sqlite_fullname)
        data = self.read_from_sqlite(sqlite_obj, table_name=data_table_name)
        sqlite_obj.commit()
        sqlite_obj.close()
        return data


def main():
    from matplotlib import pyplot as plt

    isdebug = False
    sqlite_path = "sqlites"
    mix_table_name = "mix_distr"
    uni_table_name = "uni_distr"
    scale = 0.5
    max_data = 15
    min_data = -10
    if isdebug:
        databasename = "debug"
        big_num = 3
        sample_num_expected = 30  # sample_num_expected >= (uni_part_size_expected//big_num)
        uniset_size_expected = 23  # uniset_size_expected >= big_num
        uni_part_size_expected = 52  # (sample_num_expected*big_num) >= uni_part_size_expected >= uniset_size_expected
    else:
        databasename = "distr"
        big_num = 1000  # big_num is the n2/n1
        # Make sure the following sizes fit the memory size of my computer.
        sample_num_expected = 100000  # sample_num_expected >= (uni_part_size_expected//big_num)
        uniset_size_expected = 1000000  # uniset_size_expected >= big_num
        uni_part_size_expected = 10000000  # (sample_num_expected*big_num) >= uni_part_size_expected >= uniset_size_expected
    uniset_size_base = uniset_size_expected // big_num
    uniset_size = uniset_size_base * big_num
    uni_part_size_base = uni_part_size_expected // uniset_size
    uni_part_size = uni_part_size_base * uniset_size
    part_size = uni_part_size / big_num
    sample_num_base = sample_num_expected // part_size
    sample_num = sample_num_base * part_size

    # When the database doesn't exist, create the database, which includes a Gaussian Mixture set ,
    # min_data of the Gaussian Mixture set, max_data of the Gaussian Mixture set, a normal distribution data set.
    if not os.path.exists(sqlite_path):
        os.mkdir(sqlite_path)
    sqlite_fullname = os.path.join(sqlite_path, databasename + "_sqlite.db")
    if not os.path.exists(sqlite_fullname):
        create_sqlite_distr(sqlite_fullname=sqlite_fullname,
                            mix_table_name=mix_table_name,
                            uni_table_name=uni_table_name,
                            min_data=min_data,
                            max_data=max_data,
                            sample_num=sample_num,
                            uniset_size=uniset_size,
                            big_num=big_num,
                            scale=scale,
                            isdebug=isdebug)
    else:
        print("db exists")
    
    read_sqlite_obj = read_sqlite(sqlite_fullname)
    read_mix = read_sqlite_obj(data_table_name=mix_table_name)
    mix_list=[]
    for mix_tmp in read_mix:
        mix_list.append(mix_tmp[2])
    plt.hist(mix_list, bins=100)
    plt.show()


if __name__ == '__main__':
    main()

