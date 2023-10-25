import subprocess
import os
from UitlsTools import try_mkdir


#motivation/dataset_name/Models(MLP.ResNet)/Model of different feature
def eval_all_datasets(motivation_directory:str, datasets_directory:str, feature_num:int, eval_script_file:str):
    cuda = 2
    dataset__names = [#'airfoil_self_noise', 'california_housing', 'combined_cycle_power_plant',
                      #'concrete_compressive_strength', 'physicochemical_properties_of_protein_tertiary_structure',
                      #'qsar_aquatic_toxicity', 'qsar_fish_toxicity',
                        'superconductivty_data', 'yacht_hydrodynamics']
    Model_kinds = ['MLP', 'ResNet']
    for dataset_name in dataset__names:
        dataset_dir = os.path.join(datasets_directory, dataset_name)
        for model_kind in Model_kinds:
            save_dir = os.path.join(motivation_directory, 'Data')
            try_mkdir(save_dir)
            models_dir = os.path.join(motivation_directory, dataset_name)
            models_dir = os.path.join(models_dir, model_kind)
            
            save_file_name = dataset_name+'-'+model_kind+'--'+str(feature_num)
            eval_command = f'python {eval_script_file} --dataset_dir {dataset_dir} --models_dir {models_dir} --save_dir {motivation_directory} --save_file_name {save_file_name} --feature_num {feature_num}'
            # to eval
            result = subprocess.run(eval_command, shell=True, stdout=None, stderr=None)
            # 输出命令的标准输出和标准错误
            print("命令: ", eval_command)
            # waitting for process finish

#
eval_all_datasets('./Data_motivation', './dataset/regression', 1, 'eval_from_dir_motivation.py')

