import numpy as np
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tiktoken
from sentence_transformers import SentenceTransformer
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from project.utils import *

class EssentialsFunctions:
    def __init__(self):
        self.model_embb = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def tokenizer_text(self, text, fast=True):
        if fast:
            enc = tiktoken.get_encoding("gpt2")
            tokens = enc.encode(text)
            return tokens
        else:
            tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
            tokens = tokenizer.encode(text)
            return tokens

    def embedding_text(self, text):
        embedding = self.model_embb.encode(text)
        return embedding

class EssentialDatasets:
    def __init__(self):
        self.firstClassificationData = None
        self.firstClassificationPath = None
        self.contextData = None
        self.contextPath = None

        #antes de tudo a configuração tem que rodar   
        self.load_configuration()

        self.load_context_data()
    
    def load_first_classification(self):
        with open(self.firstClassificationPath, 'r', encoding='utf-8') as f:
            self.firstClassificationData = json.load(f)
    
    def load_context_data(self):
        with open(self.contextPath, 'r', encoding='utf-8') as f:
            self.contextData = json.load(f)

    def load_configuration(self):
        with open('./project/handlerConfiguration.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.firstClassificationPath = config['paths']['traindatabase_first_class']
            self.contextPath = config["paths"]["context_path"]
    
    def open_context(self):
        conversations = []

        context = self.contextData

        for item in context:
            for key, value in item.items():
                # Divide cada fala pelo separador "|"
                parts = [x.strip() for x in value.split("|") if x.strip()]
                conversations.append(parts)

        return conversations
        
    


class FrontCortex:
    def __init__(self):
        self.essentials = EssentialsFunctions()
        self.datasets = EssentialDatasets()
        self.datasets.load_first_classification()
        self.model = None
        self.embbedindsConverteds = None
        self.tokenConverteds = None

    def Set_Model(self):
        self.conversion_Data()
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(self.embbedindsConverteds.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(len(set(self.datasets.firstClassificationData["labels"])), activation='softmax')
        ])

        self.model.compile(
            optimizer = "adam",
            loss = "sparse_categorical_crossentropy",
            metrics = ["accuracy"]
        )

    def conversion_Data(self):
        self.embbedindsConverteds = np.array([self.essentials.embedding_text(text) for text in self.datasets.firstClassificationData["text"]])
        # self.tokenConverteds = np.array([self.essentials.tokenizer_text(text) for text in self.datasets.firstClassificationData["text"]])
    

    def one_hot_encode_labels(self):
        self.encoded_labels = to_categorical(self.datasets.firstClassificationData["labels"])
    
    def train_classifier(self):
        x_train = np.array(self.embbedindsConverteds, dtype=np.float32)
        y_train = np.array(self.datasets.firstClassificationData["labels"], dtype=np.int32)

        history = self.model.fit(
            x_train, 
            y_train,
            epochs=50, 
            batch_size=32
        )
        
    def predict_class(self, text):
        embedding = self.essentials.embedding_text(text)
        embedding = np.expand_dims(embedding, axis=0)
        prediction = self.model.predict(embedding)
        predicted_class = np.argmax(prediction, axis=1)
        return predicted_class[0]
    


class MachineCortex:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        self.new_inputs_ids = None
        self. chat_history_ids = None
        self.bot_inputs_ids = None
        self.max_length = 1024
       
        #initial config
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token



    def to_think(self):
        pass

    def Dialog(self, user_input):
        self.new_inputs_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        

        if self.chat_history_ids is not None:
            self.bot_inputs_ids = torch.cat([self.chat_history_ids, self.new_inputs_ids], dim=-1)
            self.bot_inputs_ids = self.bot_inputs_ids[:, - self.max_length:]
        else:
            self.bot_inputs_ids = self.new_inputs_ids
        
        self.chat_history_ids = self.model.generate(
            self.bot_inputs_ids,
            max_length = 1000,
            pad_token_id = self.tokenizer.eos_token_id,
            do_sample = True,
            top_k = 50,
            top_p = 0.95,
            temperature = 0.7
        )

        response = self.tokenizer.decode(self.chat_history_ids[:, self.bot_inputs_ids.shape[-1]:][0], skip_special_tokens=True)
        return  translator("en","pt", response)