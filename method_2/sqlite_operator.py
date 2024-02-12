
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


class create_distr(sqlite_base):
    def __init__(self,
                 sqlite_fullname,
                 main_table_name,
                 unif_table_name,
                 min_data,
                 max_data,
                 sample_num,
                 scale,
                 isdebug):
        super().__init__(sqlite_fullname)
        self.minmax_table_name = "minmax_data"
        self.main_table_name = main_table_name
        self.unif_table_name = unif_table_name
        self.min_data = min_data
        self.max_data = max_data
        self.scale = scale
        self.isdebug = isdebug
        self.sample_num = int(sample_num)

        self.create_tables()

    def create_tables(self):
        self.create_dataset()
        self.__create_data_table(self.main_table_name)
        self.__create_data_table(self.unif_table_name)
        self.create_minmax_table()

    def create_dataset(self):
        self.distr_main = []
        self.distr_unif = []
        self.__creat_distr()

    def create_minmax_table(self):
        # Save min and max to sqlite
        sqlite_obj = sqlite3.connect(self.sqlite_fullname)
        delete_table_com = "DROP TABLE IF EXISTS %s" % self.minmax_table_name
        sqlite_obj.execute(delete_table_com)
        delete_table_com = "CREATE TABLE %s ( min_data REAL,max_data REAL )" % self.minmax_table_name
        sqlite_obj.execute(delete_table_com)

        column_list = ["min_data","max_data"]
        row_list = [(self.min_data, self.max_data)]
        self.insert_into_sqlite(sqlite_obj, table_name=self.minmax_table_name, column_list=column_list,row_list=row_list)

        sqlite_obj.commit()
        sqlite_obj.close()

    def __create_data_table(self, data_table_name):
        sqlite_obj = sqlite3.connect(self.sqlite_fullname)
        delete_table_sqlitecom = "DROP TABLE IF EXISTS %s" % data_table_name
        sqlite_obj.execute(delete_table_sqlitecom)
        create_table_sqlcom = "CREATE TABLE IF NOT EXISTS %s ( read_id INTEGER PRIMARY KEY AUTOINCREMENT, label REAL ,sample REAL )" % data_table_name
        sqlite_obj.execute(create_table_sqlcom)

        column_list, row_list = self.__create_label_data(data_table_name)
        self.insert_into_sqlite(sqlite_obj, table_name=data_table_name, column_list=column_list, row_list=row_list)

        sqlite_obj.commit()
        sqlite_obj.close()

    def __creat_distr(self):
        if self.isdebug:
            self.distr_main = list(range(self.sample_num))
            self.distr_unif = 100*np.ones(self.sample_num)
        else:
            # This is a gaussian mixture model
            for i in range(self.sample_num):
                randomint = random.randint(1, 11)
                match randomint:
                    case 1:
                        disrt_main_tmp = random.normal(loc=3, scale=self.scale)
                        while not self.min_data < disrt_main_tmp < self.max_data:
                            disrt_main_tmp = random.normal(loc=3, scale=self.scale)
                    case 2|3:
                        disrt_main_tmp = random.normal(loc=1, scale=self.scale)
                        while not self.min_data < disrt_main_tmp < self.max_data:
                            disrt_main_tmp = random.normal(loc=1, scale=self.scale)
                    case 4|5|6:
                        disrt_main_tmp = random.normal(loc=6, scale=self.scale)
                        while not self.min_data < disrt_main_tmp < self.max_data:
                            disrt_main_tmp = random.normal(loc=6, scale=self.scale)
                    case 7|8|9|10:
                        disrt_main_tmp = random.normal(loc=9, scale=self.scale)
                        while not self.min_data < disrt_main_tmp < self.max_data:
                            disrt_main_tmp = random.normal(loc=9, scale=self.scale)
                    case _ :
                        raise ValueError("randomint is wrong")
                self.distr_main.append(disrt_main_tmp)
            # Build a certain uniform distribution to get a stable result,
            # because the real uniform distribution is random, and random means unstable
            self.distr_unif = [i*(self.max_data - self.min_data)/(self.sample_num-1) + self.min_data for i in range(self.sample_num)]

    def __create_label_data(self,data_table_name):
        column_list = ['label', 'sample']
        match data_table_name:
            case self.unif_table_name:
                samples = self.distr_unif
                label = 1.0
            case self.main_table_name:
                samples = self.distr_main
                label = 0.0
            case _ :
                raise ValueError("data_table_name is wrong")
        row_list = []
        for sample in samples:
            row_list.append((label,sample))
        return column_list,row_list


class read_sqlite(sqlite_base):
    def __init__(self,sqlite_fullname):
        super().__init__(sqlite_fullname)

    def read_minmax(self):
        self.__minmax_table_name = "minmax_data"
        sqlite_obj = sqlite3.connect(self.sqlite_fullname)
        minmax_data_tmp = self.read_from_sqlite(sqlite_obj, table_name=self.__minmax_table_name)
        sqlite_obj.commit()
        sqlite_obj.close()
        min_data = minmax_data_tmp[0][0]
        max_data = minmax_data_tmp[0][1]
        return min_data,max_data

    def __call__(self,data_table_name):
        sqlite_obj = sqlite3.connect(self.sqlite_fullname)
        data = self.read_from_sqlite(sqlite_obj, table_name=data_table_name)
        sqlite_obj.commit()
        sqlite_obj.close()
        return data




