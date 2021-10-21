import torch
import torchaudio
import os
from speechbrain.pretrained import SpectralMaskEnhancement

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
)

folds = ['dev_anger_airport_0dB', 'dev_sad_airport_0dB','train_anger_airport_0dB','train_sad_airport_0dB','test_anger_airport_0dB','test_sad_airport_0dB']
in_path1 = '//home/amrgaballah/Desktop/exp_1/audio_o_r/'
out_path1 = '//home/amrgaballah/Desktop/exp_1/Human_enh/'

for fold in folds:
    in_path = os.path.join(in_path1, fold)
    print(in_path)
    out_path = os.path.join(out_path1, fold)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for filename in os.listdir(in_path):
        file1 = os.path.join(in_path, filename)
        noisy = enhance_model.load_audio(file1).unsqueeze(0)
        enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
        # Saving enhanced signal on disk
        file2 = os.path.join(out_path, filename)
        torchaudio.save(file2, enhanced.cpu(), 16000)
