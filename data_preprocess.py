import re

# more emoji need to be added
positive_emoji = ['^ __ ^', ':)', ': )', '^ ^', '^ _ ^', '^ ___ ^', ':p']
negtive_emoji = ['- __ -', ': (', ':(', ":'(", '- _ -', '- ____ -', '. _ .']

def remove_user(tweet):
    tweet = re.sub("<user>", "", tweet)
    return tweet

def replace_hashtag(tweet):
    '''
    replace hashtag
    '''
    return re.sub(r"#", "<hashtag> ", tweet)

def replace_emoticon(tweet):
    '''
    replace emoji
    '''
    tweet = re.sub('|'.join(map(re.escape, positive_emoji)), "<happy>", tweet)
    tweet = re.sub('|'.join(map(re.escape, negtive_emoji)), "<sad>", tweet)
    return tweet

def resolve_contraction(tweet):
    '''
    resolve contraction
    '''
    return re.sub(r"n\'t", ' not', tweet)

def replace_heart(tweet):
    '''
    replace heart
    '''
    tweet = re.sub("<3", "<heart>", tweet)
    tweet = re.sub("< 3", "<heart>", tweet)
    tweet = re.sub("</3", "<heartbreak>", tweet)
    tweet = re.sub("< / 3", "<heartbreak>", tweet)
    tweet = re.sub("<//3", "<heartbreak>", tweet)
    tweet = re.sub("< / / 3", "<heartbreak>", tweet)
    tweet = remove_user(tweet)
    return tweet

def replace_time(tweet):
    '''
    replace time
    '''
    return re.sub(r"[0-2][0-3]:[0-5][0-9]", ' <time> ', tweet)

def test():
    '''
    test function
    '''
    test_tweet = 'hi, i #wasn\'t dog <3 ^ ^ - __ - 11:00'

    print(preprocess(tweet))

def preprocess(tweet):

    tweet = replace_hashtag(tweet)
    tweet = replace_emoticon(tweet)
    tweet = replace_heart(tweet)
    tweet = replace_time(tweet)
    tweet = resolve_contraction(tweet)

    return tweet

def data_process(TRAIN_INPUT_PATH, TRAIN_OUTPUT_PATH, TEST_INPUT_PATH, TEST_OUTPUT_PATH):
   
    with open(TRAIN_INPUT_PATH, "r") as fp, open(TRAIN_OUTPUT_PATH, "w") as fp2:
        tweets = fp.readlines()
        for tweet in tweets:
            fp2.write(preprocess(tweet))

    with open(TEST_INPUT_PATH, "r") as fp, open(TEST_OUTPUT_PATH, "w") as fp2:
        tweets = fp.readlines()
        for tweet in tweets:
            fp2.write(tweet.split(',', 1)[0] + ',' + preprocess(tweet.split(',', 1)[1]))
