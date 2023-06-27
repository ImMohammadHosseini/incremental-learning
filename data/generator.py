
"""

"""
import numpy as np
import tensorflow as tf

class CustomGenerator(tf.keras.utils.Sequence) :
  
    def __init__(self, directory, filenames, labels, batch_size, max_len=512):
        self.directory= directory
        self.filenames= filenames
        self.labels= labels
        self.batch_size= batch_size
        self.max_len= max_len
    
    def __len__(self) :
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx) :
        batch_x = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        
        datas=[]
        for file in batch_x:
            path= self.directory+file
            with open(path) as f:
                datas.append(str(f.readlines()[0]))
                
                
        return np.array(datas), np.array(batch_y)