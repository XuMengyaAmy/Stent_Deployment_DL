from __future__ import print_function # WalabotAPI works on both Python 2 and 3.
from importlib.machinery import SourceFileLoader
import numpy as np
import time

wlbt = SourceFileLoader('WalabotAPI', 'C:/Program Files/Walabot/WalabotSDK/python/WalabotAPI.py').load_module()
wlbt.Init()

''' file name and location setting'''
fld_loc = 'D:/PhDProject/StentDetection/Dataset/D1/'
file_list = 'filelist.txt'
file_num = 276

'''
description: save file
             saves nparray in pickle format in incrementing filename
             saves filename and class in filelist
'''


def savefile(sensordata, stent_class):
    global fld_loc
    global file_num
    global file_list

    masterfile = open((fld_loc + file_list), 'a')

    if file_num < 10:
        filename = fld_loc + '00' + str(file_num)
        masterfile.write('00' + str(file_num) + ' ' + str(stent_class))
        masterfile.write("\n")
    elif file_num < 100:
        filename = fld_loc + '0' + str(file_num)
        masterfile.write('0' + str(file_num) + ' ' + str(stent_class))
        masterfile.write("\n")
    else:
        filename = fld_loc + str(file_num)
        masterfile.write(str(file_num) + ' ' + str(stent_class))
        masterfile.write("\n")

    np.save(filename, sensordata)
    masterfile.close()
    file_num = file_num + 1
    print(filename, stent_class)


'''
description: configure walabot
             Set scan profile
             Set scan range
             Set filter type
'''


def configure_wlbt():
    wlbt.Initialize()
    wlbt.ConnectAny()

    wlbt.SetProfile(wlbt.PROF_SHORT_RANGE_IMAGING)  # Set Profile - to Short-range.

    xArenaMin, xArenaMax, xArenaRes = -5, 5, 0.3  # wlbt.SetArenaX - input parameters
    yArenaMin, yArenaMax, yArenaRes = -5, 5, 0.3  # wlbt.SetArenaY - input parameters
    zArenaMin, zArenaMax, zArenaRes = 3, 12, 0.3  # wlbt.SetArenaZ - input parameters

    wlbt.SetArenaX(xArenaMin, xArenaMax, xArenaRes)
    wlbt.SetArenaY(yArenaMin, yArenaMax, yArenaRes)
    wlbt.SetArenaZ(zArenaMin, zArenaMax, zArenaRes)



    wlbt.SetDynamicImageFilter(wlbt.FILTER_TYPE_NONE)  # Walabot filtering disable


'''
description: collectdata             
             ============ command =================
             0   -   class 0
             1   -   class 1
             2   -   class 2
             3   -   class 3
             8   -   calibrate
             9   -   terminate
             ======================================
'''


def collectData():
    configure_wlbt()

    wlbt.Start()
    wlbt.SetThreshold(5)
    threshold = wlbt.GetThreshold()
    print(threshold)


    while True:
        a = input("command:")
        a = int(a)   # added. In windows, string
        wlbt.Trigger()
        wlbt.Trigger()
        imagetd, i_x, i_y, i_z, i_p = wlbt.GetRawImage()
        imagetd = np.asarray(imagetd)
        if a == 9:
            break
        elif a == 8:
            # calibrate
            wlbt.StartCalibration()
            while wlbt.GetStatus()[0] == wlbt.STATUS_CALIBRATING: wlbt.Trigger()
            continue

        elif (a == 0  or a == 1 or a == 2 or a == 3):

            # wlbt.Trigger()
            # wlbt.Trigger()
            # imagetd, i_x, i_y, i_z, i_p = wlbt.GetRawImage()
            # imagetd = np.asarray(imagetd)
            if np.all(imagetd == 0):
                print("array is all 0")
            savefile(imagetd, a)
        else:
            print("invalid command")

    wlbt.Stop()
    wlbt.Disconnect()
    wlbt.Clean()
    print('Terminate successfully')


if __name__ == '__main__':

    collectData()



