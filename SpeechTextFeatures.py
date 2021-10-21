import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer
from transformers import BertTokenizer, TFBertModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import torch
import soundfile as sf

class Speech_Recognizer:
    def __init__(self, model_name):
        # model_name from https://huggingface.co/models?pipeline_tag=automatic-speech-recognition
        self.model_name = model_name
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForMaskedLM.from_pretrained(self.model_name)

    def transcribe(self, audio_file_name):
        audio_input, sampling_rate = librosa.load(audio_file_name, sr = 16000, res_type = 'kaiser_fast')
        # audio_input, _ = sf.read(audio_file_name)
        # print(len(audio_input))
        input_values = self.tokenizer(audio_input[:int(len(audio_input)/2)], return_tensors = 'pt').input_values
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim = -1)
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]
        return transcription.lower()


class BERT_Text_Feature_Extracter:
    def __init__(self, model_name):
        # model_name from https://huggingface.co/models (bert-base-cased preferrably)
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = TFBertModel.from_pretrained(self.model_name)

    def features_fromtext(self, text):
        encoded = self.tokenizer(text, padding = 'max_length', return_tensors = 'tf')
        output = self.model(encoded['input_ids'], attention_mask = encoded['attention_mask'])
        return np.array(output.pooler_output)[0]

    def features_fromtext_batch(self, text_array, window = 50):
        data = np.array([[0]*768])

        for i in tqdm(range(0, len(text_array), window)):
            encoded = self.tokenizer(text_array[i:i+window], padding = 'max_length', return_tensors = 'tf')
            output = self.model(encoded['input_ids'], attention_mask = encoded['attention_mask'])
            temp = output.pooler_output
            data = np.concatenate([data, temp])
        
        data = data[1:]
        return data


class Speech_To_Text_Features:
    def __init__(self, mname_asr, mname_txt = None):
        #text_feature = BERT, TCNN

        self.SR = Speech_Recognizer(mname_asr)
        # if text_feature == 'BERT':
        self.TFE = BERT_Text_Feature_Extracter(mname_txt)
        # elif text_feature == 'TCNN':
            # self.TFE = TCNN_Text_Feature_Extracter()

    def get_text_features(self, audio_file_name):
        transcription = self.SR.transcribe(audio_file_name)
        txt_features = self.TFE.features_fromtext(transcription)
        return txt_features

    