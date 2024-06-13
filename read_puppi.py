#%%
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer

from src.diffmet.data.datasets import L1PFDataset
from src.diffmet.lit import LitModel
from src.diffmet.data.transforms import Compose
from src.diffmet.models import Transformer
from src.diffmet.data.datamodule import DataModule
# from src.diffmet.utils.learningcurve import make_learning_curves

from pathlib import Path
import mplhep as mh
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import vector
import torch
import yaml
vector.register_awkward()

#%%
cfg_path = './config/test-l1pf-transformer-neuron.yaml'
ckpt_path = "./lightning_logs/version_0/checkpoints/epoch=99-step=126100.ckpt"
root_path = "/scratch/x2719a04/data/L1METML/perfNano_TTbar_PU200.110X_set5.root"
#%%
with open(cfg_path, 'r') as f:
    config = yaml.safe_load(f)
model_cfg = config['model']['init_args']
augmentation = Compose(model_cfg['augmentation']['init_args']['data_list'])
preprocessing = Compose(model_cfg['preprocessing']['init_args']['data_list'])
model_args = model_cfg['model']['init_args']
model = Transformer(model_args['embed_dim'],
                    model_args['num_heads'],
                    model_args['activation'],
                    model_args['widening_factor'],
                    model_args['dropout_p'],
                    model_args['num_layers'],)

dataset_cfg = config['data']
dataset_class_path = 'src.' + dataset_cfg['dataset_class_path']
train_files = dataset_cfg['train_files']
val_files = dataset_cfg['val_files']
test_files = dataset_cfg['test_files']
best_model = LitModel.load_from_checkpoint(ckpt_path, augmentation=augmentation, preprocessing=preprocessing, model=model)
#%%
datamodule = DataModule(dataset_class_path, train_files, val_files, test_files, batch_size= 256, eval_batch_size = 512,)
#%%
predict_data = Trainer().predict(model=best_model, ckpt_path=ckpt_path, datamodule=datamodule)
#%%
predict_chunk = []
for batch in predict_data:
    for pxpy in batch:
        predict_chunk.append({'px': pxpy[0], 'py': pxpy[1]})
predict_chunk = ak.Array(data=predict_chunk, with_name='Momentum2D')
#%%
print(predict_chunk.pt)
print(len(predict_chunk.pt))



#%%
data_dict = L1PFDataset._get_data(root_path)


#%%
print(len(data_dict["baseline_chunk"].pt))
print(data_dict["baseline_chunk"].pt)
# %%
mh.style.use(mh.styles.CMS)
plt.plot(data_dict["baseline_chunk"].pt, predict_chunk.pt, ".")
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.xlabel("PUPPI MET")
plt.ylabel("RECO MET")
plt.show()
# %%
histo_range = (-300, 300)
plt.hist(data_dict["baseline_chunk"].pt - predict_chunk.pt, bins=300, range = histo_range, alpha = 0.8, label = "PUPPI - RECO")
plt.hist(data_dict["gen_met_chunk"].pt - predict_chunk.pt, bins=300, range = histo_range, alpha = 0.8, label = "Gen - RECO")
# plt.xlim(-300, 300)
plt.xlabel("Difference")
plt.legend()
plt.show()
# %%
