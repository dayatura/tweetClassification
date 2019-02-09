from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf
import pickle
import numpy

def init():

    # ============= dict =====================                                                  char_to_int

    dict_file = open("model/dict.pickle", "rb")
    
    char_to_int = pickle.load(dict_file)

    # ============== embedding ================                                                 embedding_model

    json_file = open( 'model/model_embedding.json' ,  'r' )
    loaded_model_json = json_file.read()
    json_file.close()
    
    embedding_model = model_from_json(loaded_model_json)
    embedding_model.load_weights("model/model_embedding.h5")
    embedding_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # ============= ANN =====================                                                   ann_model

    json_file = open( 'model/model.json' ,  'r' )
    loaded_model_json = json_file.read()
    json_file.close()
    ann_model = model_from_json(loaded_model_json)

    # load weights into new model
    ann_model.load_weights("model/model.h5")

    # evaluate loaded model on test data
    ann_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    graph = tf.get_default_graph()                                                              #graph
    

    # ============= Random Forest ==================                                            model_rf


    with open("model/model_RForest.pkl", 'rb') as file:  
        rf_model = pickle.load(file)
    print("Random forest model loaded")
        

    # ============= SVM ==================                                                      model_svm

    

    with open("model/model_svm.pkl", 'rb') as file:  
        svm_model = pickle.load(file)
    print("SVM model loaded")


    return char_to_int, graph, embedding_model, ann_model, rf_model, svm_model