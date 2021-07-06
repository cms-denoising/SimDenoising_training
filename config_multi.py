from magiconfig import MagiConfig
import glob

energy = 850
num_events = '*'
num_files = 10

if (num_files  % 2) == 0:
    num_files = int(num_files/2)
else:
    num_files = int((num_files-1)/2)

def get_files(filetype):
    if filetype == 'Fuzz':
        list_of_names = []
        for name in glob.glob('/storage/local/data1/gpuscratch/leiningt/ntup_photon_energy'+str(energy)+'*Prod*_n'+str(num_events)+'_part*.root'):
            list_of_names.append(name)
        list_of_names.sort()
        list_of_names_t = list_of_names[:num_files]
        list_of_names_v = list_of_names[num_files:]
        " ".join(list_of_names_t)
        " ".join(list_of_names_v)
    if filetype == 'Sharp':
        list_of_names = []
        for name in glob.glob('/storage/local/data1/gpuscratch/leiningt/ntup_photon_energy'+str(energy)+'*phi0.0_n'+str(num_events)+'_part*.root'):
            list_of_names.append(name)
        list_of_names.sort()
        list_of_names_t = list_of_names[:num_files]
        list_of_names_v = list_of_names[num_files:]
        " ".join(list_of_names_t)
        " ".join(list_of_names_v)
    return list_of_names_t[0], list_of_names_v[0]

fuzzy_t_files, fuzzy_v_files = get_files('Fuzz') 
sharp_t_files, sharp_v_files = get_files('Sharp') 

config = MagiConfig()
config.batchSize = 50
config.epochs = 100
config.features = 100
config.kernelSize = 3
config.lr = 0.001
config.num_layers = 9
config.num_workers = 8
config.outf = 'out-train-aug8-2'
config.transform = 'normalize'
config.patchSize = 100
config.sigma = 20
config.trainfileFuzz = fuzzy_t_files
config.trainfileSharp = sharp_t_files
config.valfileFuzz = fuzzy_v_files
config.valfileSharp = sharp_v_files 
