from magiconfig import MagiConfig
import glob

energy = 700
num_events = 200
num_files = 2

def get_files(filetype):
    if filetype == 'Fuzz':
        list_of_names = []
        for name in glob.glob('ntup_photon_energy'+str(energy)+'*Prod*_n'+str(num_events)+'.root'):
            list_of_names.append(name)
        list_of_names.sort()
        list_of_names = list_of_names[:num_files]
        " ".join(list_of_names)
    if filetype == 'Sharp':
        list_of_names = []
        for name in glob.glob('ntup_photon_energy'+str(energy)+'*phi0.0_n'+str(num_events)+'.root'):
            list_of_names.append(name)
        list_of_names.sort()
        list_of_names = list_of_names[:num_files]
        " ".join(list_of_names)
    return list_of_names[0]

config = MagiConfig()
config.batchSize = 40
config.epochs = 50
config.features = 9
config.kernelSize = 3
config.lr = 0.001
config.num_layers = 9
config.num_workers = 8
config.outf = 'out-train-jul20-1'
config.transform = 'normalize'
config.patchSize = 20
config.sigma = 20
config.trainfileFuzz = get_files('Fuzz')
config.trainfileSharp = get_files('Sharp')
config.valfileFuzz = get_files('Fuzz')
config.valfileSharp = get_files('Sharp')