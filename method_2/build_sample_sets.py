import os
import sqlite_operator as sql
from matplotlib import pyplot as plt

isdebug = False
sqlite_path = "sqlites"
main_table_name = "main_distr"
unif_table_name = "unif_distr"
scale = 0.5 # for the gaussian mixture model
# The interval [l,r] of the uniform distribution = [min_data,max_data]
max_data = 20
min_data = -10
if isdebug:
    databasename = "debug"
    sample_num = 30
else:
    databasename = "distr"
    sample_num = 500000


def main():
    if not os.path.exists(sqlite_path):
        os.mkdir(sqlite_path)
    sqlite_fullname = os.path.join(sqlite_path, databasename + "_sqlite.db")
    sql.create_distr(
                     sqlite_fullname=sqlite_fullname,
                     main_table_name=main_table_name,
                     unif_table_name=unif_table_name,
                     min_data=min_data,
                     max_data=max_data,
                     sample_num=sample_num,
                     scale=scale,
                     isdebug=isdebug)

    read_sqlite_obj = sql.read_sqlite(sqlite_fullname)
    read_main = read_sqlite_obj(data_table_name=main_table_name)
    main_list = []
    for main_tmp in read_main:
        main_list.append(main_tmp[2])
    plt.hist(main_list, bins=100)
    plt.show()


if __name__ == '__main__':
    main()