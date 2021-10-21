import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
import os


enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir="pretrained_models/mtl-mimic-voicebank",)

folds = ['dev_anger_airport_0dB', 'dev_sad_airport_0dB','train_anger_airport_0dB','train_sad_airport_0dB','test_anger_airport_0dB','test_sad_airport_0dB']
in_path1 = '//home/amrgaballah/Desktop/exp_1/audio_o_r/'
out_path1 = '//home/amrgaballah/Desktop/exp_1/Machine_enh/'

for fold in folds:
    in_path = os.path.join(in_path1, fold)
    print(in_path)
    out_path = os.path.join(out_path1, fold)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for filename in os.listdir(in_path):
        file1 = os.path.join(in_path, filename)
    
        enhanced = enhance_model.enhance_file(file1)
        # Saving enhanced signal on disk
        file2 = os.path.join(out_path, filename)
    
        torchaudio.save(file2, enhanced.unsqueeze(0).cpu(), 16000)
