"""

TEST FILE TO TRAIN MODEL.

In general, this file should serve as a guideline of 
how to train model with the outrageclf package.


Positional Arguments:
    - filepath: file name of training csv file e.g. "./*.csv"
    - savepath: path to where models are saved
    - filename: name of saved models
    - model: specify one of the model architect to train e.g: "LSTM", "GRU"
    - text_column: name of training text column e.g. "text"
    - class_column: name of class column e.g "outrage"

Available model:
    - LSTM with Glove Twitter
    - GRU with Glove Twitter

"""


import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import keras
import outrageclf
from outrageclf.preprocessing import WordEmbed, get_lemmatize_hashtag, create_embedding_matrix_default
from outrageclf.model_architect import lstm_model, deep_gru_model


model = ["LSTM", "GRU"]

if __name__ == '__main__':
    #Initialize the parser
    parser = argparse.ArgumentParser(description="Outrage Classifier Training. Developed by The Crockett Lab")
    parser.add_argument(
        "filepath",
        help='specifying the path to the training dataset. This should be in the form of .../*.csv'
    )

    parser.add_argument(
        "savepath",
        help= ('specifying the path to save the model.',
            'There will be two files being saved to this path: a tokenizer and a trained model'
            )
    )

    parser.add_argument(
        "filename",
        help= ('name of the training file.',
            'This is used to attached to the name of tokenizer and the trained model.')
    )

    parser.add_argument(
        "model",
        help= 'specifying the model for the training. Default value is "LSTM". Allowed values are '+', '.join(model),
        choices=model,
        nargs='?',
        default="LSTM",
        metavar = "MODEL"
    )

    parser.add_argument(
        "text_column",
        help= 'name of text column in csv file'
    )

    parser.add_argument(
        "class_column",
        help= 'name of class column in csv file. This must be in the form of binary 0, 1 data type'
    )


    args = parser.parse_args()
    df = pd.read_csv(args.filepath)
    print ("File loaded")

    word_embed = WordEmbed()
    tokenizer_path = args.savepath + args.filename + '_tokenizer' + '.joblib'
    lemmatize_hashtag = get_lemmatize_hashtag(df[args.text_column])
    # train the new tokenizer and the embedding matrix for the model
    word_embed._train_new_tokenizer(lemmatize_hashtag, tokenizer_path)
    word_index = word_embed.tokenizer.word_index
    embedding_matrix = create_embedding_matrix_default(word_index)
    print ("Embedding matrix created")
    
    # get X and y train
    X_train = word_embed._get_embedded_vector(lemmatize_hashtag)
    y_train = np.array(df[args.class_column])
    print ("Training data prepared")

    if args.model == 'LSTM':
        model = lstm_model(
            embedding_matrix,
            vocab_size = len(word_index) + 1
        )
    elif args.model == 'GRU':
        model = deep_gru_model(
            embedding_matrix,
            vocab_size = len(word_index) + 1
        )
    
    # train model
    history = model.fit(
        X_train,
        y_train,
        epochs = 20,
        batch_size = 300,
        verbose = 1
    )
    
    # save model
    model_path = args.savepath + args.filename + '.h5'
    model.save(model_path)

    print("Finish training and write " + args.model + " model to:" + model_path)
