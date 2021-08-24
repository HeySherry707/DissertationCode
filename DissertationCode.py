#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np




import tweepy





import botometer





idFile = open("tweet_id.txt", "r")




idLines = idFile.readlines()





idList = []
for tweetID in idLines:
    tweetID = tweetID.strip();
    idList.append(tweetID)




df = pd.DataFrame({'tweetID':idList})




auth = tweepy.AppAuthHandler("xxx", "xxx")
api = tweepy.API(auth, retry_count=5, retry_delay=2, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)





responseDF = pd.DataFrame(columns=['tweetID', 'userID', 'text','language','screen_name','name','followers_count','favorite_count', 'retweet_count'])




a = 0
b = 100
for i in range(0,5700,1):
    print("This is Loop %d, processing tweet_id number from %d to %d" %(i,a,b))
    idRange = range(a,b,1)
    tmpList=[]
    for j in idRange:
        tmpList.append(idList[j])
    print("The length of tmpList is %d" %(len(tmpList)))
    response = api.statuses_lookup(tmpList, tweet_mode="extended", trim_user=False)
    for status in response:
        c = len(responseDF)
        print("The length of reponseDF is %d" %(c))
        try:
            full_text = status.retweeted_status.full_text
        except AttributeError: # Not a Retweet
            full_text = status.full_text
        responseDF.loc[c] = [status.id, status.user.id, full_text, status.lang, status.user.screen_name, status.user.name, status.user.followers_count,status.favorite_count, status.retweet_count]
    a+=100
    b+=100





responseDF.drop_duplicates(subset=['tweetID',"text"], inplace=True, ignore_index=True)




wholeDF = responseDF[responseDF['text'].str.contains('climate change is real|climate change is false|climate change is fake|climate change not real|climate change hoax|global warming is real|global warming is false|global warming is fake|global warming not real|global warming hoax|climatechangeisreal|climatechangeisfalse|climatechangeisfake|climatechangenotreal|climatechangehoax|globalwarmingisreal|globalwarmingisfalse|globalwarmingisfake|globalwarmingnotreal|globalwarminghoax', case=False)]



wholeDF = wholeDF.reset_index(drop=True)




wholeDF = wholeDF[wholeDF['language']=='en']




userDF = pd.DataFrame(wholeDF, columns=["userID"])



userDF.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)



accounts = userDF["userID"].to_list()




rapidapi_key = 'xxx'
twitter_app_auth = {
    'consumer_key': 'xxx',
    'consumer_secret': 'xxx',
    'access_token': 'xxx',
    'access_token_secret': 'xxx',
  }




bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)




sanityDF = pd.DataFrame(columns=['userID', 'cap_en','overall','astroturf','fake_follower','financial','spammer','self_declared','other','majority_lang'])




sanityID = ['xxx','xxx']



for id, result in bom.check_accounts_in(sanityID):
    Len = len(sanityDF)
    try:
        sanityDF.loc[Len]=[str(id), result['cap']['english'],                             result['raw_scores']['english']['overall'],                             result['raw_scores']['english']['astroturf'],                            result['raw_scores']['english']['fake_follower'], result['raw_scores']['english']['financial'],                             result['raw_scores']['english']['spammer'],                             result['raw_scores']['english']['self_declared'],                             result['raw_scores']['english']['other'],                             result['user']['majority_lang']]
    except Exception as e:
        pass
    continue




sanityID2 = ['xxx','xxx','xxx']




for id, result in bom.check_accounts_in(sanityID2):
    Len = len(sanityDF)
    try:
        sanityDF.loc[Len]=[str(id), result['cap']['english'],                             result['raw_scores']['english']['overall'],                             result['raw_scores']['english']['astroturf'],                            result['raw_scores']['english']['fake_follower'], result['raw_scores']['english']['financial'],                             result['raw_scores']['english']['spammer'],                             result['raw_scores']['english']['self_declared'],                             result['raw_scores']['english']['other'],                             result['user']['majority_lang']]
    except Exception as e:
        pass
    continue




sanityDF.to_csv('sanityCheck.csv', index=False, header=True)


scoreDF = pd.DataFrame(columns=['userID', 'cap_en','overall','astroturf','fake_follower','financial','spammer','self_declared','other'])



pd.set_option('display.float_format',lambda x : '%.3f' % x)



for id, result in bom.check_accounts_in(account):
    scoreLen = len(scoreDF)
    try:
        scoreDF.loc[scoreLen]=[str(id), result['cap']['english'],                             result['raw_scores']['english']['overall'],                             result['raw_scores']['english']['astroturf'],                            result['raw_scores']['english']['fake_follower'], result['raw_scores']['english']['financial'],                             result['raw_scores']['english']['spammer'],                             result['raw_scores']['english']['self_declared'],                             result['raw_scores']['english']['other']]
    except Exception as e:
        pass
    continue



scoreDF.drop_duplicates(subset=['userID'], inplace=True, ignore_index=True)




scoreDF = scoreDF.reset_index(drop=True)



scoreDF.to_csv('scoreDF_noDup.csv', index=False, header=True)



wholeDF[['userID']] = wholeDF[['userID']].astype(str)




joinedDF = wholeDF.join(scoreDF.set_index('userID'), on = 'userID')



joinedDF = joinedDF.dropna(subset=['overall', 'cap_en','astroturf'])



joinedDF.reset_index(drop=True)


joinedDF.loc[(joinedDF['astroturf']>=0.5) & (joinedDF['cap_en']>=0.80),'AstroturfBot'] = 'Bot'


joinedDF.loc[(joinedDF['astroturf']< 0.5) | (joinedDF['cap_en']<0.80),'AstroturfBot'] = 'non-Bot'



joinedDF.AstroturfBot.value_counts()



joinedDF.loc[joinedDF['text'].str.contains('climate change is real|global warming is real|climatechangeisreal|globalwarmingisreal', case=False), 'AcceptOrDeny'] = 'Accept'



joinedDF.loc[joinedDF['text'].str.contains('climate change is false|climate change is fake|climate change not real|climate change hoax|global warming is false|global warming is fake|global warming not real|global warming hoax|climatechangeisfalse|climatechangeisfake|climatechangenotreal|climatechangehoax|globalwarmingisfalse|globalwarmingisfake|globalwarmingnotreal|globalwarminghoax', case=False), 'AcceptOrDeny'] = 'Deny'


joinedDF.AcceptOrDeny.value_counts()



# joinedDF.to_csv('joinedDF_0815.csv', index=False, header=True)




testBD = joinedDF[(joinedDF['AstroturfBot']=='Bot') & (joinedDF['AcceptOrDeny']=='Deny')]



testBD['tweetID'] = testBD['tweetID'].astype('str')



testBD['userID'] = testBD['userID'].astype('str')


testBD['tweetID'].describe()



testBD['userID'].describe()



joinedDF['tweetID'] = joinedDF['tweetID'].astype('str')




joinedDF['userID'] = joinedDF['userID'].astype('str')


denyAstroturf0815 = joinedDF[(joinedDF['AstroturfBot']=='Bot') & (joinedDF['AcceptOrDeny']=='Deny')]



# filteredBD is the data set which include the mannually verified attitude and comments about the tweets which was preliminarily labeled as 'Deny'
filteredBD = pd.read_excel('denyAstroturf_manualLabelled.xlsx', index_col=None)




filteredBD.reset_index(drop=True)



filteredBD.drop(['tweetID','userID'],axis=1,inplace=True)



filteredBD.insert(0,'tweetID',testBD['tweetID'].values.tolist())



filteredBD.insert(1,'userID',testBD['userID'].values.tolist())



filteredBD.to_csv('manualLabeled.csv', header=True, index=False)



# Only falsely labeled rows are kept
filteredBDfalse = filteredBD[filteredBD['Verified Attitude']=='Accept']



falseList = filteredBDfalse.tweetID.values.tolist()



len(falseList)



joinedDF.loc[joinedDF['tweetID'].isin(falseList), 'AcceptOrDeny'] = 'Accept'



# The dataset with the texts causing fault classification is re-classified
joinedDF.loc[joinedDF['text'].str.contains('xxxxxx', case=False), 'AcceptOrDeny'] = 'Accept'



joinedDF = joinedDF.reset_index(drop=True)


joinedDF.to_csv("Dataset.csv", header=True, index=False)


from scipy.stats import anderson



from scipy.stats import spearmanr



anderson(joinedDF['astroturf'],dist='norm')



anderson(joinedDF['self_declared'],dist='norm')



anderson(joinedDF['fake_follower'],dist='norm')


spearmanr(joinedDF['astroturf'],joinedDF['self_declared'])



spearmanr(joinedDF['astroturf'],joinedDF['fake_follower'])


astroturfDF.self_declared.describe()


anderson(joinedDF['polarity'],dist='norm')



anderson(joinedDF['subjectivity'],dist='norm')



spearmanr(joinedDF['astroturf'],joinedDF['polarity'])



spearmanr(joinedDF['astroturf'],joinedDF['subjectivity'])



acceptAstroturf = joinedDF[(joinedDF['AstroturfBot']=='Bot') & (joinedDF['AcceptOrDeny']=='Accept')]



acceptAstroturf = acceptAstroturf.reset_index(drop=True)


len(acceptAstroturf)



denyAstroturf = joinedDF[(joinedDF['AstroturfBot']=='Bot') & (joinedDF['AcceptOrDeny']=='Deny')]



denyAstroturf = denyAstroturf.reset_index(drop=True)



len(denyAstroturf)



denyCommon = joinedDF[(joinedDF['AstroturfBot']=='non-Bot') & (joinedDF['AcceptOrDeny']=='Deny')]



denyCommon = denyCommon.reset_index(drop=True)


len(denyCommon)



acceptCommon = joinedDF[(joinedDF['AstroturfBot']=='non-Bot') & (joinedDF['AcceptOrDeny']=='Accept')]



acceptCommon = acceptCommon.reset_index(drop=True)



len(acceptCommon)



len(joinedDF)


denyAll = joinedDF[joinedDF['AcceptOrDeny']=='Deny']



denyAll = denyAll.reset_index(drop=True)



len(denyAll)




acceptAll = joinedDF[joinedDF['AcceptOrDeny']=='Accept']


acceptAll = acceptAll.reset_index(drop=True)


len(acceptAll)


joinedDF.AstroturfBot.value_counts()


astroturfDF = joinedDF[joinedDF['AstroturfBot']=='Bot']


astroturfDF = astroturfDF.reset_index(drop=True)



astroturfTotalUser = len(astroturfDF['userID'].drop_duplicates())



TotalUser = len(joinedDF['userID'].drop_duplicates())


TotalUser



len(astroturfDF)



len(astroturfDF)/len(joinedDF)


astroturfTotalUser


len(astroturfDF)/astroturfTotalUser



commonTotalDF = joinedDF[joinedDF['AstroturfBot']=='non-Bot']



commonTotalUser = len(commonTotalDF['userID'].drop_duplicates())



commonTotalUser



len(commonTotalDF)



len(commonTotalDF)/commonTotalUser



print(TotalUser)



print(astroturfTotalUser)



TotalAstroturfRatio = astroturfTotalUser / TotalUser


print('There are %d astroturfing bot accounts, accounting for %f of %d total users' %(astroturfTotalUser,TotalAstroturfRatio,TotalUser))



acceptAstroturfUser = len(acceptAstroturf['userID'].drop_duplicates())



len(acceptAstroturf)



acceptAstroturfRatio = acceptAstroturfUser / astroturfTotalUser


acceptAstroturfTotalRatio = acceptAstroturfUser / TotalUser



print('There are %d astroturfing bot accounts accepting climate change, accounting for %f of %d total astroturfing users, accounting for %f of all %d users' %(acceptAstroturfUser,acceptAstroturfRatio,astroturfTotalUser,acceptAstroturfTotalRatio,TotalUser))



denyAstroturfUser = len(denyAstroturf['userID'].drop_duplicates())



denyAstroturfRatio = denyAstroturfUser / astroturfTotalUser



denyAstroturfTotalRatio = denyAstroturfUser / TotalUser



print('There are %d astroturfing bot accounts denying climate change, accounting for %f of %d total astroturfing users, accounting for %f of all %d users' %(denyAstroturfUser,denyAstroturfRatio,astroturfTotalUser,denyAstroturfTotalRatio,TotalUser))



acceptCommonUser = len(acceptCommon['userID'].drop_duplicates())



acceptCommonRatio = acceptCommonUser / commonTotalUser
acceptCommonTotalRatio = acceptCommonUser / TotalUser

print('There are %d common accounts accepting climate change, accounting for %f of %d total common users, accounting for %f of all %d users' %(acceptCommonUser,acceptCommonRatio,commonTotalUser,acceptCommonTotalRatio,TotalUser))



denyCommonUser = len(denyCommon['userID'].drop_duplicates())




denyCommonRatio = denyCommonUser / commonTotalUser
denyCommonTotalRatio = denyCommonUser / TotalUser

print('There are %d common accounts denying climate change, accounting for %f of %d total common users, accounting for %f of all %d users' %(denyCommonUser,denyCommonRatio,commonTotalUser,denyCommonTotalRatio,TotalUser))



acceptAllUser = len(acceptAll['userID'].drop_duplicates())




acceptAllUser




denyAllUser = len(denyAll['userID'].drop_duplicates())



denyAllUser




print(acceptAstroturfUser)



print(denyAstroturfUser)




astroturfDF.astroturf.describe()




astroturfDF.self_declared.describe()




astroturfDF.fake_follower.describe()




from textblob import TextBlob



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



joinedDF['polarity'] = joinedDF['text'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)



joinedDF['subjectivity'] = joinedDF['text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)



from numpy import mean, ptp, var, std




joinedDF.polarity.describe()




std(joinedDF.polarity)




plt.hist(joinedDF.polarity, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.title('Polarity Distribution of All Tweets (TT)')
plt.show()



joinedDF.subjectivity.describe()



std(joinedDF.subjectivity)




plt.hist(joinedDF.subjectivity, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('Subjectivity Score')
plt.ylabel('Frequency')
plt.title('Subjectivity Distribution of All Tweets (TT)')
plt.show()




acceptAstroturf.polarity.describe()




std(acceptAstroturf.polarity)



plt.hist(acceptAstroturf.polarity, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.title('Polarity Distribution of Astroturfing Tweets Accepting Climate Change (BA)')
plt.show()




acceptAstroturf.subjectivity.describe()




std(acceptAstroturf.subjectivity)



plt.hist(acceptAstroturf.subjectivity, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('Subjectivity Score')
plt.ylabel('Frequency')
plt.title('Subjectivity Distribution of Astroturfing Tweets Accepting Climate Change (BA)')
plt.show()




denyAstroturf.polarity.describe()


std(denyAstroturf.polarity)



plt.hist(denyAstroturf.polarity, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.title('Polarity Distribution of Astroturfing Tweets Denying Climate Change (BD)')
plt.show()



denyAstroturf.subjectivity.describe()



std(denyAstroturf.subjectivity)



plt.hist(denyAstroturf.subjectivity, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('Subjectivity Score')
plt.ylabel('Frequency')
plt.title('Subjectivity Distribution of Astroturfing Tweets Denying Climate Change (BD)')
plt.show()



acceptCommon.polarity.describe()



std(acceptCommon.polarity)



plt.hist(acceptCommon.polarity, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.title('Polarity Distribution of Non-Astroturfing Tweets Accepting Climate Change (CA)')
plt.show()



acceptCommon.subjectivity.describe()



std(acceptCommon.subjectivity)




plt.hist(acceptCommon.subjectivity, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('Subjectivity Score')
plt.ylabel('Frequency')
plt.title('Subjectivity Distribution of Non-Astroturfing Tweets Accepting Climate Change (CA)')
plt.show()



denyCommon.polarity.describe()



std(denyCommon.polarity)



plt.hist(denyCommon.polarity, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.title('Polarity Distribution of Non-Astroturfing Tweets Denying Climate Change (CD)')
plt.show()



denyCommon.subjectivity.describe()



std(denyCommon.subjectivity)



plt.hist(denyCommon.subjectivity, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('Subjectivity Score')
plt.ylabel('Frequency')
plt.title('Subjectivity Distribution of Non-Astroturfing Tweets Denying Climate Change (CD)')
plt.show()




commonTotalDF.polarity.describe()




commonTotalDF.subjectivity.describe()




astroturfDF.polarity.describe()



astroturfDF.subjectivity.describe()



acceptAll.polarity.describe()



acceptAll.subjectivity.describe()



denyAll.polarity.describe()




denyAll.subjectivity.describe()




acceptAstroturf.favorite_count.describe()



acceptAstroturf.favorite_count.sum()



acceptAstroturf.followers_count.describe()



acceptAstroturf.followers_count.sum()



denyAstroturf.favorite_count.describe()




denyAstroturf.favorite_count.sum()




denyAstroturf.followers_count.describe()




denyAstroturf.followers_count.sum()




commonTotalDF.favorite_count.describe()



acceptCommon.followers_count.describe()




acceptCommon.followers_count.sum()


acceptCommon.favorite_count.sum()



acceptCommon.favorite_count.describe()




denyCommon.followers_count.describe()



denyCommon.followers_count.sum()



denyCommon.favorite_count.describe()



denyCommon.favorite_count.sum()



astroturfDF.favorite_count.describe()



astroturfDF.favorite_count.sum()



astroturfDF.followers_count.describe()



astroturfDF.followers_count.sum()



commonTotalDF.favorite_count.describe()



commonTotalDF.favorite_count.sum()



commonTotalDF.followers_count.describe()



commonTotalDF.followers_count.sum()



import re



import nltk; nltk.download('stopwords')



from pprint import pprint



import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel



import spacy



nlp = spacy.load('en_core_web_sm')



import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)




import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



import pyLDAvis



import pyLDAvis.gensim_models




pyLDAvis.enable_notebook()



from nltk.corpus import stopwords




stop_words = stopwords.words('english')





def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out




dataTT = joinedDF.text.values.tolist()
dataTT = [re.sub('\S*@\S*\s?', '', sent) for sent in dataTT]
dataTT = [re.sub("\_", " ", sent) for sent in dataTT]
dataTT = [re.sub("(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:\/~\+#]*[\w\-\@?^=%&amp;\/~\+#])?", "", sent) for sent in dataTT]
dataTT = [re.sub("https", " ", sent) for sent in dataTT]
dataTT = [re.sub('\s+', ' ', sent) for sent in dataTT]
dataTT = [re.sub("\'", "", sent) for sent in dataTT]
dataTT = [re.sub("\&amp", "", sent) for sent in dataTT]
data_wordsTT = list(sent_to_words(dataTT))
bigram = gensim.models.Phrases(data_wordsTT, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_wordsTT], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
data_wordsTT_nostops = remove_stopwords(data_wordsTT)
data_wordsTT_bigrams = make_bigrams(data_wordsTT_nostops)
dataTT_lemmatized = lemmatization(data_wordsTT_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
id2word = corpora.Dictionary(dataTT_lemmatized)
texts = dataTT_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]
coherence_list = []
model_list = []
for num_topic in range(2,30,2):
    lda_modelTT = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topic,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_modelTT, texts=dataTT_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_list.append(coherence_lda)
    model_list.append(lda_modelTT)
    print('\nNumber of topics: %d, Coherence Score: %f'  %(num_topic,coherence_lda))
x = range(2,30,2)
plt.plot(x, coherence_list)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.title("Choosing optimal number of topics (TT)")
plt.show()




lda_modelTT = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=12,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
pprint(lda_modelTT.print_topics())
doc_lda = lda_modelTT[corpus]
vis = pyLDAvis.gensim_models.prepare(lda_modelTT, corpus, id2word)
vis



dataBA = acceptAstroturf.text.values.tolist()
dataBA = [re.sub('\S*@\S*\s?', '', sent) for sent in dataBA]
dataBA = [re.sub("\_", " ", sent) for sent in dataBA]
dataBA = [re.sub("(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:\/~\+#]*[\w\-\@?^=%&amp;\/~\+#])?", "", sent) for sent in dataBA]
dataBA = [re.sub("https", " ", sent) for sent in dataBA]
dataBA = [re.sub("http", " ", sent) for sent in dataBA]
dataBA = [re.sub('\s+', ' ', sent) for sent in dataBA]
dataBA = [re.sub("\'", "", sent) for sent in dataBA]
dataBA = [re.sub("\&amp", "", sent) for sent in dataBA]
data_wordsBA = list(sent_to_words(dataBA))
bigram = gensim.models.Phrases(data_wordsBA, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_wordsBA], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
data_wordsBA_nostops = remove_stopwords(data_wordsBA)
data_wordsBA_bigrams = make_bigrams(data_wordsBA_nostops)
dataBA_lemmatized = lemmatization(data_wordsBA_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
id2word = corpora.Dictionary(dataBA_lemmatized)
texts = dataBA_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]
coherence_list = []
for num_topic in range(2,20,2):
    lda_modelBA = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topic,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_modelBA, texts=dataBA_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_list.append(coherence_lda)
    print('\nNumber of topics: %d, Coherence Score: %f'  %(num_topic,coherence_lda))
x = range(2,20,2)
plt.plot(x, coherence_list)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.title("Choosing optimal topic number for astroturfing bots accepting climate change (BA)")
plt.show()



lda_modelBA = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=8,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
pprint(lda_modelBA.print_topics())
doc_lda = lda_modelBA[corpus]
vis = pyLDAvis.gensim_models.prepare(lda_modelBA, corpus, id2word)
vis




dataCA = acceptCommon.text.values.tolist()
dataCA = [re.sub('\S*@\S*\s?', '', sent) for sent in dataCA]
dataCA = [re.sub("\_", " ", sent) for sent in dataCA]
dataCA = [re.sub("(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:\/~\+#]*[\w\-\@?^=%&amp;\/~\+#])?", "", sent) for sent in dataCA]
dataCA = [re.sub("https", " ", sent) for sent in dataCA]
dataCA = [re.sub('\s+', ' ', sent) for sent in dataCA]
dataCA = [re.sub("\'", "", sent) for sent in dataCA]
dataCA = [re.sub("\&amp", "", sent) for sent in dataCA]
data_wordsCA = list(sent_to_words(dataCA))
bigram = gensim.models.Phrases(data_wordsCA, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_wordsCA], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
data_wordsCA_nostops = remove_stopwords(data_wordsCA)
data_wordsCA_bigrams = make_bigrams(data_wordsCA_nostops)
dataCA_lemmatized = lemmatization(data_wordsCA_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
id2word = corpora.Dictionary(dataCA_lemmatized)
texts = dataCA_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]
coherence_list = []
for num_topic in range(2,20,2):
    lda_modelCA = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topic,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_modelCA, texts=dataCA_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_list.append(coherence_lda)
    print('\nNumber of topics: %d, Coherence Score: %f'  %(num_topic,coherence_lda))
x = range(2,20,2)
plt.plot(x, coherence_list)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.title("Choosing optimal topic number for non-astroturfing tweets accepting climate change (CA)")
plt.show()



lda_modelCA = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=12,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
pprint(lda_modelCA.print_topics())
doc_lda = lda_modelCA[corpus]
vis = pyLDAvis.gensim_models.prepare(lda_modelCA, corpus, id2word)
vis



dataCD = denyCommon.text.values.tolist()
dataCD = [re.sub('\S*@\S*\s?', '', sent) for sent in dataCD]
dataCD = [re.sub("\_", " ", sent) for sent in dataCD]
dataCD = [re.sub("(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:\/~\+#]*[\w\-\@?^=%&amp;\/~\+#])?", "", sent) for sent in dataCD]
dataCD = [re.sub("https", " ", sent) for sent in dataCD]
dataCD = [re.sub('\s+', ' ', sent) for sent in dataCD]
dataCD = [re.sub("\'", "", sent) for sent in dataCD]
dataCD = [re.sub("\&amp", "", sent) for sent in dataCD]
data_wordsCD = list(sent_to_words(dataCD))
bigram = gensim.models.Phrases(data_wordsCD, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_wordsCD], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
data_wordsCD_nostops = remove_stopwords(data_wordsCD)
data_wordsCD_bigrams = make_bigrams(data_wordsCD_nostops)
dataCD_lemmatized = lemmatization(data_wordsCD_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
id2word = corpora.Dictionary(dataCD_lemmatized)
texts = dataCD_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]
coherence_list = []
for num_topic in range(2,20,1):
    lda_modelCD = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topic,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_modelCD, texts=dataCD_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_list.append(coherence_lda)
    print('\nNumber of topics: %d, Coherence Score: %f'  %(num_topic,coherence_lda))
x = range(2,20,1)
plt.plot(x, coherence_list)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.title("Choosing optimal topic number for non-astroturfing tweets denying climate change (CD)")
plt.show()



lda_modelCD = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=5,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
pprint(lda_modelCD.print_topics())
doc_lda = lda_modelCD[corpus]
vis = pyLDAvis.gensim_models.prepare(lda_modelCD, corpus, id2word)
vis



dataBD = denyAstroturf.text.values.tolist()
dataBD = [re.sub('\S*@\S*\s?', '', sent) for sent in dataBD]
dataBD = [re.sub("\_", " ", sent) for sent in dataBD]
dataBD = [re.sub("(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:\/~\+#]*[\w\-\@?^=%&amp;\/~\+#])?", "", sent) for sent in dataBD]
dataBD = [re.sub("https", " ", sent) for sent in dataBD]
dataBD = [re.sub('\s+', ' ', sent) for sent in dataBD]
dataBD = [re.sub("\'", "", sent) for sent in dataBD]
dataBD = [re.sub("\&amp", "", sent) for sent in dataBD]
data_wordsBD = list(sent_to_words(dataBD))
bigram = gensim.models.Phrases(data_wordsBD, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_wordsBD], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
data_wordsBD_nostops = remove_stopwords(data_wordsBD)
data_wordsBD_bigrams = make_bigrams(data_wordsBD_nostops)
dataBD_lemmatized = lemmatization(data_wordsBD_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
id2word = corpora.Dictionary(dataBD_lemmatized)
texts = dataBD_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]
coherence_list = []
for num_topic in range(1,7,1):
    lda_modelBD = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topic,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_modelBD, texts=dataBD_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_list.append(coherence_lda)
    print('\nNumber of topics: %d, Coherence Score: %f'  %(num_topic,coherence_lda))
x = range(1,7,1)
plt.plot(x, coherence_list)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.title("Choosing optimal topic number for astroturfing bots denying climate change (BD)")
plt.show()




lda_modelBD = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=3,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
pprint(lda_modelBD.print_topics())
doc_lda = lda_modelBD[corpus]
vis = pyLDAvis.gensim_models.prepare(lda_modelBD, corpus, id2word)
vis



lda_modelBD1 = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=1,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
pprint(lda_modelBD1.print_topics())
doc_lda = lda_modelBD1[corpus]
vis = pyLDAvis.gensim_models.prepare(lda_modelBD1, corpus, id2word)
vis
