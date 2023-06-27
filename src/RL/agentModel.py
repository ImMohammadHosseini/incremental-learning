
"""

"""


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

class AgentDeepModel(Model):
    def __init__(
        self, 
        class_num:int,
        
    ):
        super().__init__()
        
        self.dense1 = Dense(8, input_dim=(class_num,), activation='relu')
        self.dense2 = Dense(2, activation='sigmoid')

    def call(self, inputs):         
        x = self.dense1(inputs)
        return self.dense2(x)
 