'''
Author: qyp422
Date: 2023-02-06 00:49:08
Email: qyp422@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2023-02-13 14:59:50
Description: 

Copyright (c) 2023 by qyp422, All Rights Reserved. 
'''
import os,sys,datetime,time
import dumpreader

def main():
    if len(sys.argv)< 5:
        sys.exit('Syntax: $> python get_datafile.py input_filename top_filename output_filename framenum detalbp(circleoption)')
    pwd = os.getcwd() 
    filename = os.path.basename(sys.argv[1])
    topfile = os.path.basename(sys.argv[2])
    output_filename = sys.argv[3]
    framenum = int(sys.argv[4])
    try:
        detalbp = int(sys.argv[5])
    except:
        detalbp = 0
    r = dumpreader.Lammps_dumpreader(filename,topfile)
    # if change seq
    n = r._system._N
    s = 'C'*(n//2)+'G'*(n//2)
    r._system.change_seq(seq=s)

    #----------------------
    if r.output_timestep_datafile(framenum,output_filename=output_filename,deltabp = detalbp):
        print('already done!')
    else:
        print('fail!')

if __name__ == "__main__":
        main()