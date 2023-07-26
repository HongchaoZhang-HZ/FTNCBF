from SNCBF_Synth import *

class FTNCBF():
    '''
    Define intiial FT-NCBF framework, the overall NCBF looks like follows:
    Sensor Faults - FTEst(EKFs) - SNCBFs - Conflict Resolution (Mini-Norm Controller)
    '''
    def __init__(self, arch, act_layer, case, num_sensors, Faults, verbose=False):
        '''

        :param arch:
        :param act_layer:
        :param case:
        :param num_sensors:
        :param Faults:
        :param verbose:
        '''
        NCBF_list = []

        # Define SNCBFs
        SNCBF = SNCBF_Synth(arch, act_layer, case, verbose=verbose)

        # Define SNCBFs' EKF Gain
        # Todo: update SNCBF EKF Gain

        # Train SNCBFs
        SNCBF.train(num_epoch=10, num_restart=0, warm_start=False)

