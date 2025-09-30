from pathlib import Path
import pyjson5

def load_dataset_configs(configs_path):
    assert Path(configs_path).exists(), \
            print(f'The given config file ({configs}) does not exist!')
        
    configs = pyjson5.load(open(configs_path, 'r')) #TODO add validation for these configs
    
    # Get data sources and GSDs
    tmp = configs['dataset_type'].split('_')
    # format: "sen2_xx_mod_yy"
    gsd = {tmp[0]: tmp[1], tmp[2]: tmp[3]}

    # Update the configs with the SEN2 or MODIS bands to be used
    data_source = configs['datasets']['data_source']
    for band in configs['datasets']['selected_bands'][data_source].keys():
        configs['datasets']['selected_bands'][data_source][band] = configs['datasets'][f'{data_source}_bands'][gsd[data_source]][band]

    return configs