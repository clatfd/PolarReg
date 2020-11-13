from sys import platform
if platform == 'win32':
    DESKTOPdir = '//Desktop4'
    #DATADESKTOPdir = '//Desktop2'
    DATADESKTOPdir = 'T:/'
    taskdir = DESKTOPdir+'/Dtensorflow\LiChen\VW\PolarReg'
    MODELDESKTOPdir = '//Desktop2'
else:
    #ubuntu
    DESKTOPdir = '/mnt/desktop4'
    #DATADESKTOPdir = '/mnt/desktop2'
    DATADESKTOPdir = '/mnt/V'
    taskdir = '/home/li/pycharm'
    MODELDESKTOPdir = '/mnt/desktop2'
