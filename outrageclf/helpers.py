import emoji, re, collections, string
import nltk
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords


# top emojis
# the list is practically derived from our datasets
top_emojis = ['😂','🤣','😡','🖕','😹','🙏','👎','🌊','🙄','🤔']
lemmatizer = WordNetLemmatizer()
cachedStopWordsPunctuation = set(stopwords.words("english")
                                 + [x for x in list(string.punctuation) if x not in ['!','?']]
                                 + ['',' ','  '])

# check if emojis in a string
def char_is_emoji(char):
    return char in emoji.UNICODE_EMOJI['en']

s = set(emoji.UNICODE_EMOJI['en'].values())
def string_is_emoji_name(text):
    return text in s

# extract a string of emojis from a string
def extract_emojis(text):
    return ' '.join(c for c in text if c in emoji.UNICODE_EMOJI)

# just get the hashtag
# this function removes the function, even in hashtag
def get_hashtag(text):
    text = re.sub(r'[%s]' % re.escape("""!"$%&()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)
    return ",".join([i.lower()  for i in text.split() if i.startswith("#") ])

# word_tokenize as defined in nltk library
def token_postag(text):
    tokens = word_tokenize(text)
    return pos_tag(tokens)

# function to simplify POS
# exclusively used for lemmatization using WORDNET
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# similary to get_wordnet_pos
# but more granular in order to create more features
def modify_pos(dict):
    result_dic = {}
    for key in dict.keys():
        if key.startswith('J'):
            if "adj" in result_dic:
                result_dic["adj"] += dict[key]
            else:
                result_dic["adj"] = dict[key]
        elif key.startswith('V'):
            if "verb" in result_dic:
                result_dic["verb"] += dict[key]
            else:
                result_dic["verb"] = dict[key]
        elif key.startswith('N'):
            if "noun" in result_dic:
                result_dic["noun"] += dict[key]
            else:
                result_dic["noun"] = dict[key]
        elif key.startswith('R'):
            if "adv" in result_dic:
                result_dic["adv"] += dict[key]
            else:
                result_dic["adv"] = dict[key]
        elif key in ['PRP', 'PRP$']:
            if "pronoun" in result_dic:
                result_dic["pronoun"] += dict[key]
            else:
                result_dic["pronoun"] = dict[key]
        elif key.startswith('W'):
            if "wh" in result_dic:
                result_dic["wh"] += dict[key]
            else:
                result_dic["wh"] = dict[key]
        else:
            if "other" in result_dic:
                result_dic["other"] += dict[key]
            else:
                result_dic["other"] = dict[key]
    return result_dic


# tokenize and then stemmize the string
def token_stem_lemmatize(text):
    tokens_pos = token_postag(text)
    result_string = ''
    for word, tag in tokens_pos:
        wntag = get_wordnet_pos(tag)
        # not return tag in case of None
        if wntag is None:
            result_string += lemmatizer.lemmatize(word.lower())
        else:
            result_string += lemmatizer.lemmatize(word.lower(), pos=wntag)
        result_string += ' '
    return result_string

# remove stop words and short words
def stop_short_process(text):
    text = ' '.join([word for word in text.split() if word not in cachedStopWordsPunctuation])
    text = re.sub("[^a-zA-Z ]+", '', text) # remove apostrophe now
    text = ' '.join(word for word in text.split() if len(word)>2)
    return text

# wrap up over the processing
def tweet_process(tweet):
    tweet = re.sub('//t.co\S+', ' ', tweet) # remove link
    tweet = re.sub('http\S+\s*', ' ', tweet)  # remove URLs
    tweet = re.sub('@\S+', ' ', tweet)  # remove mentions
    tweet = re.sub('&amp', ' ', tweet)  # remove mentions
    tweet = re.sub('RT@|RT @', ' ', tweet)  # remove RT
    tweet = re.sub('#\S+', ' ', tweet)  # remove hashtags
    tweet = re.sub('[%s]' % re.escape("""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', tweet)  # remove punctuations, leave behind apostrophe"'"
    tweet = re.sub('\s+', ' ', tweet)  # remove extra whitespace
    tweet = token_stem_lemmatize(tweet)
    tweet = stop_short_process(tweet)
    return tweet

# check if the tweet has embedded link
def has_link(tweet):
    short_link = re.findall('//t.co\S+',tweet)
    url_link = re.findall('http\S+\s*',tweet)
    result = 0 if not short_link and not url_link else 1
    return (result)


def psy_tweet_process(tweet):
    stemmer = SnowballStemmer("english")
    tokenizer = TweetTokenizer()
    tweet_tokenized = tokenizer.tokenize(tweet)
    n = len(tweet_tokenized)
    try:
        tweet_tokenized = [unicode(y.encode("utf-8"), errors='ignore') for y in tweet_tokenized]
        stemmed = [stemmer.stem(y) for y in tweet_tokenized]
    except:
        stemmed = [stemmer.stem(y) for y in tweet_tokenized]
        stemmed = [d for d in stemmed if d not in cachedStopWordsPunctuation]
    return stemmed, n
