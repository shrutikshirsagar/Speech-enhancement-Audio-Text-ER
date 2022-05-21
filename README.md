![ser_AT5 (1)](https://user-images.githubusercontent.com/34964872/169665226-d5aa1754-b045-4649-8870-395917f5dfae.png)


#### In this pipleine, Step 1: we first get the enhancement audio files from the pretrained models from the speechbrain
#### Step 2: we get the speech transcripts and BERT feature based on the transcripts
#### step 3: we extract modulation feature and we convert enhancemnet audio to 16 bit in order to extract egemap fetaures
#### step 4: Finally, we run neural network weher we first trained the model and saved it so that we can use the same trained model for tetsing different noise types and noise leveles. 


Chnage the path and run enh_pipeline.sh script
