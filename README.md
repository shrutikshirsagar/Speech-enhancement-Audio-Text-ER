Chnage the path and run enh_pipeline.sh script

#### In this pipleine, Step 1: we first get the enhancement audio files from the pretrained models from the speechbrain
#### Step 2: we get the speech transcripts and BERT feature based on the transcripts
#### step 3: we extract modulation feature and we convert enhancemnet audio to 16 bit in order to extract egemap fetaures
#### step 4: Finally, we run neural network weher we first trained the model and saved it so that we can use the same trained model for tetsing different noise types and noise leveles. 

conda activate speechbrain
cd /home/amrgaballah/Desktop/enhancement/
python resample_all.py
python MELD_matricgan_all.py
cd /home/amrgaballah/Desktop/enhancement/ASR_enhance/
python MELD_asrlossenh_all.py




conda activate ASR_BERT
cd /home/amrgaballah/Desktop/exp_1/SER/utils/
python save_transcripts_all.py
python save_to_tree_all.py



conda activate base
cd /home/amrgaballah/Documents/MSF/
python modspec_avec.py
cd /home/amrgaballah/Desktop/exp_1/
python 16bit_MELD_conversion.py
cd /home/amrgaballah/Documents/Opensmile/opensmile-2.3.0/
python egemaps_feat_ext.py


conda activate ASR_BERT
cd /home/amrgaballah/Desktop/exp_1/SER/audio_text/
python text_audio_train.py
python text_audio_test_all.py
