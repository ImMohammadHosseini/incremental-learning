
"""

"""
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class StudentModel(Model):
    def __init__(
        self, 
        preprossesor, 
        bert_layer,
        class_num,
        max_len:int = 512,
        
    ):
        super().__init__()
        self.bert_inputs= preprossesor
        self.bert_layer = bert_layer
        
        self.dense1 = Dense(16, activation='relu')
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(8, activation='relu')
        self.dropout2 = Dropout(0.2)
        self.out = Dense(class_num, activation='softmax')
        
    def call(self, inputs):
        x = self.bert_inputs(inputs)
        x = self.bert_layer(x)
        x = x['pooled_output']

        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.out(x)
