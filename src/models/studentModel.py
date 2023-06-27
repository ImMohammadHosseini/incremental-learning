
"""

"""
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def student_model(preprossesor, bert_layer, max_len=512): 
    inp= Input(shape=(), dtype=tf.string)
    bert_inputs= preprossesor(inp)
    outputs = bert_layer(bert_inputs)#pooled_output, sequence_output
    x= outputs['pooled_output']

    x= Dense(16, activation='relu')(x)
    x= Dropout(0.2)(x)
    x= Dense(8, activation='relu')(x)
    x= Dropout(0.2)(x)
    out= Dense(class_num, activation='softmax')(x)

    model= Model(inputs=inp, outputs=out)
    model.compile(Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model