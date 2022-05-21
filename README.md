![ser_AT5 (1)](https://user-images.githubusercontent.com/34964872/169665226-d5aa1754-b045-4649-8870-395917f5dfae.png)  <br />
 Fig 1: Experimental pipeline for AER using audio and text features.

 
Step 1: we first get the enhancement audio files from the pretrained models from the speechbrain. <br />
Step 2: we get the speech transcripts and BERT feature based on the transcripts. <br />
step 3: we extract modulation feature and we convert enhancemnet audio to 16 bit in order to extract egemap fetaures. <br />
step 4: Finally, we run neural network weher we first trained the model and saved it so that we can use the same trained model for tetsing different noise types and noise leveles. <br />


In order to run the above proposed approached described in fig 1, Chnage the path and run enh_pipeline.sh script.  <br />

![specogram](https://user-images.githubusercontent.com/34964872/169665581-0d0155ea-2d67-4fc3-a790-22ed5cc9d1c9.png)  <br />
Fig 2: Modulation spectrogram for different conditions, from top to bottom: clean, (airport) noisy at 0 dB , MetriGan+, and
mimic-loss ennhanced speech. Left plots correspond to angry emotion and right plots to sad emotio

