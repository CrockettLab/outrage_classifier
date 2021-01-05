from .model_architect import threshold_acc
from .preprocessing import WordEmbed, get_lemmatize_hashtag
from keras.models import load_model
from joblib import load


'''
Load pretrained model

Input: url
    - Users responsibility to acquire the h5 model format
    - and input correct url link

Output: 
'''
def _load_crockett_model(url):
    return load_model(
        url,
        custom_objects={'threshold_acc': threshold_acc}
    )

'''
Wrapper function for prediction:
In general, if users have to call this function several times
it it more efficient to load the model and use built-in predict method.

Input: text vector, lemmatize_url, model_url
'''
def pretrained_model_predict(text_vector, lemmatize_url, model_url):
    word_embed = WordEmbed()
    word_embed._get_pretrained_tokenizer(lemmatize_url)
    model = _load_crockett_model(model_url)

    lemmatized_text = get_lemmatize_hashtag(text_vector)
    embedded_vector = word_embed._get_embedded_vector(lemmatized_text)
    predict = model.predict(embedded_vector)

    return pretrained_model_predict
