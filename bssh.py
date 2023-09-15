'''
Author: qyp422
Date: 2023-09-14 10:36:38
Email: qyp422@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2023-09-14 20:44:20
Description: 

Copyright (c) 2023 by qyp422, All Rights Reserved. 
'''

bashname = '4'
dir = r'/share/home/myqiang2/arraybu/mixed27/'

infile_total = 20
every_file = 4

core_every = 128//every_file
file_n = infile_total//every_file

for i in range(file_n):
    with open(bashname+f'{i}.sh', 'w',newline='\n') as f:
        f.write(r'#!/bin/bash'+'\n')
        f.write(r'cd '+ dir + '\n')
        for k in range(every_file): 
            t_num = i*every_file+k
            f.write(f'(mpirun -np {core_every} lmp21 -in in.lk_T_300_{t_num} >& {t_num}.out )&\n')
    print(f'bsub -J {bashname}{i} -n 128 -q amd_milan bash {bashname}{i}.sh')