
"""

"""



import tensorflow as tf


#_________________________________________________________________________-
#GPU test
cpu = tf.config.experimental.list_physical_devices('CPU')[0]
gpu = tf.config.experimental.list_physical_devices('GPU')
if len(gpu)>0:
    gpu = gpu[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    #tf.config.experimental.set_visible_devices(cpu)
    print("GPU known")
else :
    print("GPU unknown")

#_________________________________________________________________________-