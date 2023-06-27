
"""
"""
import os


def split_data_points(directory, subset):
    
    new_path= directory+'/'+'all'+subset
    os.mkdir(new_path)
    file_num=0
    with open(directory+'/train.csv') as f:
        lines = f.readlines()
        for i in range(1, len(lines), 2):
            file_num+=1
            with open(new_path+'/'+lines[i+1][2]+'_'+str(file_num)+'.txt', 'a+') as f1:
                f1.write(lines[i])
        '''for l in lines:
            file_num+=1
            with open(new_path+'/'+l[9]+'_'+str(file_num)+'.txt', 'a+') as f1:
                f1.write(l[11:])'''
                                

#_________________________________________________________________________-

def filenames_and_labels(directory, subset):
    directory= directory+'/'+'all'+subset
    filenames= os.listdir(directory)
    labels=[]
    for name in filenames:
        labels.append(int(name[0])-1)
        
    #labels= label_binarize(labels,classes=[1,2])
    #labels= [to_categorical(i-1, num_classes =class_num, dtype='int32') for i in labels] 
    #data= zip(filenames, labels)
    return filena