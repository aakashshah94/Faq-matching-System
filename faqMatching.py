import time
import nltk
import csv
import pysolr
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tag.util import tuple2str
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.parse.stanford import StanfordDependencyParser as sdp
from nltk.parse.corenlp import CoreNLPDependencyParser
import copy
import operator
from nltk import pos_tag


nltkWnMap = {
    'NN': wn.NOUN,
    'VB': wn.VERB,
    'JJ': wn.ADJ,
    'RB': wn.ADV,
    'NNP': wn.NOUN
    }
cnlp = CoreNLPDependencyParser()
STANFORD_VER = 'stanford-corenlp-3.9.1'
PATH_TO_JAR = 'jars/stanford-corenlp-full-2018-02-27/' + STANFORD_VER + '.jar'
PATH_TO_MODELS_JAR = 'jars/stanford-corenlp-full-2018-02-27/' + STANFORD_VER + '-models.jar'
STANFORD_DEP_PARSER = sdp(path_to_jar=PATH_TO_JAR, path_to_models_jar=PATH_TO_MODELS_JAR)


custom_syn_map={
'encryption' : 'cryptography',
'blockchain' : 'ledger',
'mining' : 'minting',
'wallet' : 'account',
'mint' : 'create',
'finite' : 'limited',
'computer' : 'laptop',
'regulate' : 'rules',
}
custom_hyper_map={'bitcoin': ['cryptocurrency'],
                  'cryptocurrency': ['currency'],
                  'cryptography': ['mathematics'],
                  'encryption': ['mathematics'],
                  'blockchain': ['record'],
                  'signature': ['encryption', 'trust'],
                  'mining': ['coin'],
                  'exchange': ['bank'],
                  'wallet': ['ledger'],
                  'financial crimes enforcement network': ['law enforcement'],
                  'united states treasury department': ['law enforcement'],
                  'bubble': ['illusion'],
                  'multisig': ['signature'],
                  'proof of work': ['verification'],
                  'block': ['blockchain']}

custom_hyper_map_reverse={
'cryptocurrency' : ['bitcoin'],
'currency' : ['cryptocurrency'],
'mathematics' : ['cryptography','encryption'],
'record' : ['blockchain'],
'encryption' : ['signature'],
'trust' : ['signature'],
'coin' : ['mining'],
'bank' : ['exchange'],
'ledger' : ['wallet'],
'law enforcement' : ['financial crimes enforcement network','united states treasury department'],
'illusion' : ['bubble'],
'signature' : ['multisig'],
'verification' : ['proof of work'],
'blockchain' : ['block'],
}
featureMaxLimit = 30
stopWords = set(stopwords.words('english'))


def get_list_of_questions():
	questions = []
	questions.append(
		("How does mining in bitcoin work?","How does Bitcoin mining work?"))
	questions.append(
		("Are there any disadvantages?","What are the disadvantages of Bitcoin?"))
	questions.append(
		("Is bitcoin a fraudulent system?","Is Bitcoin a Ponzi scheme?"))
	questions.append(
		("how do i accept bitcoin as a payment for my business?","How to Accept bitcoin payments for your store?"))
	questions.append(
		("what happens when my laptop is turned down?","what if i receive a bitcoin when my computer is powered off?"))
	questions.append(
		("Is bitcoin going to crash?","Won't Bitcoin fall in a deflationary spiral?"))
	questions.append(
		("Who owns the Bitcoin?","Who controls the Bitcoin?"))
	questions.append(
		("What can I buy with Bitcoin?","What Can You get With Bitcoin?"))
	questions.append(
		("Who invented Bitcoin?","Who created Bitcoin?"))
	questions.append(
		("How does bitcoin effect my taxes?","What about Bitcoin and taxes?"))
	return questions


def mean_reciprical_rank(positions):
	mrr = 0
	qlen = len(positions)

	for p in positions:
		if p != 0:
			mrr = mrr + 1/float(p)
	mrr = 1/float(qlen)*mrr

	return mrr


def correct_pos(question, top_10, gold=None, original_question=None):
	q = ""

	if original_question:
		q = original_question
	else:
		q = question

	pos = 1
	for t in top_10:
		if t == q:
			return pos
		pos = pos + 1
	return 0


def run_model(question,csv_data,task=2):
	top_10 = None
	if task == 2:
		top_10 = task2(csv_data,question[0])
	elif task == 5:
         question1 = nltk.word_tokenize(question[0])
         top_10 = searchintask4(question1)
	pos = 0
	for p, t in enumerate(top_10):
		if t.lower() == question[1].lower():
			pos = p+1
	return pos

def is_correct(test, correct):
	if test == correct:
		return True
	return False

def run_analysis(csv_data,task=2):
	question_pair_lists = get_list_of_questions()
	positions = []
	for q in question_pair_lists:
		pos = run_model(q,csv_data,task)
		positions.append(pos)
	mrr = mean_reciprical_rank(positions)
	print ("Correct positions = {0}".format(positions))
	print ("MRR: {0}".format(mrr))



def read_data(file):
    csv_data = []
    with open(file, 'r',  encoding="utf8") as csvfile:
        d = csv.reader(csvfile, delimiter=',', quotechar='"')
        for r in d:
            csv_data.append( (r[0].lower(),r[1].replace('\n','').lower()))
    return csv_data


def task2(csv_data,question):
    bag_of_words = tokenize(csv_data)
    question_dict,words = getInputAndTokenize(question)
    result = calculatOverlap(question_dict,bag_of_words,words)
    rank = 1
    prev = result[0][1]
    top_10 = []

    print ("#1"+str(csv_data[result[0][0]]))
    top_10.append(str(csv_data[result[0][0]][0]))
    for i in range(1,10):
         if result[i][1] < prev:
            rank += 1
            prev = result[i][1]
         print ("#"+str(rank)+str(csv_data[result[i][0]]))
         top_10.append("#"+str(rank)+str(csv_data[result[i][0]][0]))
    return top_10


def task2_default(csv_data):
    bag_of_words = tokenize(csv_data)
    name = input("Input Question: ").lower()
    question_dict,words = getInputAndTokenize(name)
    result = calculatOverlap(question_dict,bag_of_words,words)
    rank = 1
    prev = result[0][1]
    top_10=[]

    print ("#1"+str(csv_data[result[0][0]][0]))
    top_10.append(str(csv_data[result[0][0]][0]))
    for i in range(1,10):
         if result[i][1] < prev:
            rank += 1
            prev = result[i][1]
         #print ("#"+str(rank)+str(csv_data[result[i][0]]))
         print ("#"+str(rank)+str(csv_data[result[i][0]][0]))
         top_10.append("#"+str(rank)+str(csv_data[result[i][0]][0]))
    return top_10


def indexingProcessTask2(csv_data):
    solrInstance = 'http://localhost:8983/solr/new_core/'
    start = time.time()
    solr = pysolr.Solr(solrInstance)
    counter = 0
    data = []
    stemmer = PorterStemmer()
    questionId = 1
    for tuple in csv_data:
        line = tuple[0].strip()
        tokensque = word_tokenize(line)
        tokensans = word_tokenize(tuple[1])
        data.append({
                'id': str(questionId),
                'faq': ' '.join(tokensque),
                'ans': ' '.join(tokensans),
            })
        questionId += 1
    if data:
        solr.add(data)
    print(time.time() - start)

def searchSolrTask2():
    solrInstance = 'http://localhost:8983/solr/new_core/'
    solr = pysolr.Solr(solrInstance)
    # query = '&'.join(query.split())
    # q_list = []
    # if query:
    #     q_list.append('text:' + query)
    # q_string = ', '.join(q_list)
    q_list = []
    q_list.append('faq:' + "created^5.0")
    q_list.append('faq:' + "who")
    q_list.append('faq:' + "bitcoin")
    q_list.append('faq:' + "?")
    q_string = ', '.join(q_list)
    result = solr.search(q=q_string, fl='*, score')

    print()
    print("------------------")
    print("| SEARCH RESULTS |")
    print("------------------")
    print()
    print("Saw {0} result(s).".format(len(result)))
    j = 0
    for result1 in result:
        j += 1
        print(j)
        # print(result1)
        temp = result1['id']
        art = str(temp).split("_")
        # print("Article : " + art[0])
        # print("Sentence : " + art[1])
        print(result1['score'])
        print(result1['faq'])
        print("-----------------------")


def searchintask4(query):
    solrInstance = 'http://localhost:8983/solr/task3/'
    solr = pysolr.Solr(solrInstance)
    stemmer = PorterStemmer()
    tokensTaggedTest = pos_tag(query)
    head_words = dependencyRel(' '.join(query))
    synonymsTest, hypernymsTest, hyponymsTest, meronymsTest, holonymsTest = getFeatures(tokensTaggedTest, query)
    lemmasTest = getLemmas(tokensTaggedTest)
    stem11 = [stemmer.stem(t) for t in query]
    stemTest = ' '.join(stem11)
    listTagged = [tuple2str(t) for t in tokensTaggedTest]
    posData = ' '.join(listTagged)
    posData = '&'.join(posData.split())
    lemmasTest = '&'.join(lemmasTest.split())
    stemTest = '&'.join(stemTest.split())
    synonymsTest = '&'.join(synonymsTest.split())
    hypernymsTest = '&'.join(hypernymsTest.split())
    hyponymsTest = '&'.join(hyponymsTest.split())
    holonymsTest = '&'.join(holonymsTest.split())
    meronymsTest = '&'.join(meronymsTest.split())
    head_words = '&'.join(head_words.split())
    q_list = []
    if query:
        q_list.append('faq:' + ' '.join(query) + '^4.8')
        q_list.append('faq_ans'+' '.join(query)+'^2.8')
    if posData:
        q_list.append('pos_tag_q:' + posData + '^0.02')
        q_list.append('pos_tag_a:' + posData + '^0.001')
    if lemmasTest:
        q_list.append('lemma_q:' + lemmasTest + '^3.0')
        q_list.append('lemma_a:' + lemmasTest + '^2.0')
    # if stemTest:
    #     q_list.append('stem:' + stemTest + '^1.5')
    if synonymsTest:
        q_list.append('synonyms_q:' + synonymsTest + '^3.0')
        q_list.append('synonyms_a:' + synonymsTest + '^1.5')
    if hypernymsTest:
        q_list.append('hypernyms_q:' + hypernymsTest + '^3.0')
        q_list.append('hypernyms_a:' + hypernymsTest + '^1.5')
    if head_words:
        q_list.append('head_words_q:' + head_words + '^3.0')
        q_list.append('head_words_a:' + head_words + '^2.0')
    if hyponymsTest:
         q_list.append('hyponyms_q:' + hyponymsTest + '^0.24')
         q_list.append('hyponyms_a:' + hyponymsTest + '^0.14')
    if meronymsTest:
         q_list.append('meronyms_q:' + meronymsTest + '^0.14')
         q_list.append('meronyms_a:' + meronymsTest + '^0.10')
    if holonymsTest:
         q_list.append('holonyms_q:' + holonymsTest + '^0.14')
         q_list.append('holonyms_a:' + holonymsTest + '^0.10')
    # print(','.join(q_list))
    q_string = ', '.join(q_list)
    print("Query is: ")
    print("q="+q_string+", fl='*, score', rows="+str(10))
    input("Press Enter to continue...")

    result = solr.search(q=q_string, fl='*, score', rows = 10)
    # for r in json.dumps(result.docs):
    #     print(r)
    # for r in result:
    #     print(r['id'], r['text'])
    #     # print(r['text'])
    print()
    print("------------------")
    print("| SEARCH RESULTS |")
    print("------------------")
    print()
    print("Saw {0} result(s).".format(len(result)))
    j = 0
    top_10=[]
    for result1 in result:
        j += 1
        print(j)
        # print(result1)
        temp = result1['id']
        art = str(temp).split("_")
        # print("Article : " + art[0])
        # print("Sentence : " + art[1])
        sen = result1['faq_original']
        print(sen)
        top_10.append(sen[0])
        print(result1['score'])
        print("-----------------------")
    print(str(top_10[0]))
    return top_10



def searchintask4_default(query):
    solrInstance = 'http://localhost:8983/solr/task3/'
    solr = pysolr.Solr(solrInstance)
    stemmer = PorterStemmer()
    tokensTaggedTest = pos_tag(query)
    head_words = dependencyRel(' '.join(query))
    synonymsTest, hypernymsTest, hyponymsTest, meronymsTest, holonymsTest = getFeatures(tokensTaggedTest, query)
    lemmasTest = getLemmas(tokensTaggedTest)
    stem11 = [stemmer.stem(t) for t in query]
    stemTest = ' '.join(stem11)
    listTagged = [tuple2str(t) for t in tokensTaggedTest]

    posData = ' '.join(listTagged)
    posData = '&'.join(posData.split())
    lemmasTest = '&'.join(lemmasTest.split())
    stemTest = '&'.join(stemTest.split())
    synonymsTest = '&'.join(synonymsTest.split())
    hypernymsTest = '&'.join(hypernymsTest.split())
    hyponymsTest = '&'.join(hyponymsTest.split())
    holonymsTest = '&'.join(holonymsTest.split())
    meronymsTest = '&'.join(meronymsTest.split())
    head_words = '&'.join(head_words.split())
    q_list = []
    if query:
        q_list.append('faq:' + ' '.join(query) + '^1.8')
        q_list.append('faq_ans'+' '.join(query)+'^0.8')
    if posData:
        q_list.append('pos_tag_q:' + posData + '^0.02')
        q_list.append('pos_tag_a:' + posData + '^0.001')
    if lemmasTest:
        q_list.append('lemma_q:' + lemmasTest + '^2.0')
        q_list.append('lemma_a:' + lemmasTest + '^1.0')
    # if stemTest:
    #     q_list.append('stem:' + stemTest + '^1.5')
    if synonymsTest:
        q_list.append('synonyms_q:' + synonymsTest + '^3.0')
        q_list.append('synonyms_a:' + synonymsTest + '^1.5')
    if hypernymsTest:
        q_list.append('hypernyms_q:' + hypernymsTest + '^4.0')
        q_list.append('hypernyms_a:' + hypernymsTest + '^3.5')
    if head_words:
        q_list.append('head_words_q:' + head_words + '^3.0')
        q_list.append('head_words_a:' + head_words + '^2.0')
    if hyponymsTest:
         q_list.append('hyponyms_q:' + hyponymsTest + '^0.24')
         q_list.append('hyponyms_a:' + hyponymsTest + '^0.14')
    if meronymsTest:
         q_list.append('meronyms_q:' + meronymsTest + '^0.14')
         q_list.append('meronyms_a:' + meronymsTest + '^0.10')
    if holonymsTest:
         q_list.append('holonyms_q:' + holonymsTest + '^0.14')
         q_list.append('holonyms_a:' + holonymsTest + '^0.10')
    # print(','.join(q_list))
    q_string = ', '.join(q_list)
    print("Query is: ")
    print("q="+q_string+", fl='*, score', rows="+str(10))
    input("Press Enter to continue...")

    result = solr.search(q=q_string, fl='*, score', rows = 10)
    # for r in json.dumps(result.docs):
    #     print(r)
    # for r in result:
    #     print(r['id'], r['text'])
    #     # print(r['text'])
    print()
    print("------------------")
    print("| SEARCH RESULTS |")
    print("------------------")
    print()
    print("Saw {0} result(s).".format(len(result)))
    j = 0
    top_10=[]
    for result1 in result:
        j += 1
        print(j)
        # print(result1)
        temp = result1['id']
        art = str(temp).split("_")
        # print("Article : " + art[0])
        # print("Sentence : " + art[1])
        sen = result1['faq']
        print(sen)
        top_10.append(sen[0])
        print(result1['score'])
        print("-----------------------")
    print(str(top_10[0]))
    return top_10

def tokenize(csv_data):
    bag_of_words = []
    for c in csv_data:
        question = {}
        answer = {}
        for x in nltk.word_tokenize( c[0].lower()):
            if x in question:
             question[x] = question[x]+1
            else:
             question[x] = 1
        for x in nltk.word_tokenize(c[1].lower()):
            if x in answer:
                answer[x] = answer[x]+1
            else:
                answer[x] = 1
        bag_of_words.append((question,answer))
    return bag_of_words

def getInputAndTokenize(name):
    words = nltk.word_tokenize(name)
    question_dict = {}

    for x in words:
        if x in question_dict:
            question_dict[x] = question_dict[x]+ 1
        else:
            question_dict[x] = 1
    print(question_dict)
    return question_dict,words

def calculatOverlap(question_dict,bag_of_words,words):
    result =[]
    i = 0
    for x,y in bag_of_words:
         if x.keys() == question_dict.keys():
            result.append((i , len(words)))
         else:
            count = 0
            temp_copy = copy.deepcopy(question_dict)
            for j in temp_copy:
                if j in x:
                     if temp_copy[j] > x[j]:
                         temp_copy[j] = temp_copy[j] - x[j]
                         count += x[j]
                     else:
                         count += temp_copy[j]
                         temp_copy[j] = 0
            for j in temp_copy:
                 if j in y:
                     if temp_copy[j] > y[j]:
                         temp_copy[j] = temp_copy[j] - y[j]
                         count += y[j]
                     else:
                        count += temp_copy[j]
                        temp_copy[j] = 0
            result.append((i,count))
         i += 1
    result = sorted(result,key=operator.itemgetter(1), reverse=True)
    return result


def dependencyRel(line):
    # result = STANFORD_DEP_PARSER.raw_parse(line)
    result = cnlp.raw_parse(line)
    # parse_result = STANFORD_PARSER.raw_parse(line)
    dep_tree = [r for r in result]
    # parse_tree = [r for r in parse_result]
    dep_dict = dep_tree[0]
    # head = dep_dict.root['word']
    head1 = set()
    for head, rel, dep in dep_dict.triples():
        head1.add(head[0])
    return ' '.join(head1)



def indexingProcessTask3(csv_data):
    solrInstance = 'http://localhost:8983/solr/task3/'
    start = time.time()
    solr = pysolr.Solr(solrInstance)
    stemmer = PorterStemmer()
    data = []
    questionid = 1
    for tuple in csv_data:
        tokensque = word_tokenize(tuple[0].strip())
        stopRemovedQue = removestop(tokensque)
        tokensans = word_tokenize(tuple[1].strip())
        stopRemovedans = removestop(tokensans)
        postaggedque = pos_tag(tokensque)
        postaggedans = pos_tag(tokensans)
        list_pos_tagged_que=[tuple2str(t) for t in postaggedque]
        list_pos_tagged_ans=[tuple2str(t) for t in postaggedans]
        lemma1 = getLemmas(postaggedque)
        head_words_que = dependencyRel(tuple[0].strip())
        head_words_ans = dependencyRel(tuple[1].strip())
        extra_heads_que = set()
        for x in head_words_que.split():
            if x in custom_syn_map:
                extra_heads_que.add(custom_syn_map[x])
        for x in extra_heads_que:
            head_words_que += " "+x

        stem1 = [stemmer.stem(t) for t in tokensque]
        lemma2 = getLemmas(postaggedans)
        stem2 = [stemmer.stem(t) for t in tokensans]
        synonyms, hypernyms, hyponyms, meronyms, holonyms = getFeatures(postaggedque, tokensque)
        synonyms_a, hypernyms_a, hyponyms_a, meronyms_a, holonyms_a = getFeatures(postaggedans, tokensans)
        data.append({
                    'id':str(questionid) ,
                    'faq_original': str(tuple[0]),
                    'faq': ' '.join(tokensque),
                    'stop_words_q': ' '.join(stopRemovedQue),
                    'pos_tag_q': ' '.join(list_pos_tagged_que),
                    'lemma_q': lemma1,
                    'stem_q': ' '.join(stem1),
                    'head_words_q': head_words_que,
                    'synonyms_q': synonyms,
                    'hypernyms_q': hypernyms,
                    'hyponyms_q': hyponyms,
                    'meronyms_q': meronyms,
                    'holonyms_q': holonyms,
                    'faq_ans': ' '.join(tokensans),
                    'stop_words_a': ' '.join(stopRemovedans),
                    'pos_tag_a': ' '.join(list_pos_tagged_ans),
                    'lemma_a': lemma2,
                    'stem_a': ' '.join(stem2),
                    'head_words_a': head_words_ans,
                    'synonyms_a': synonyms_a,
                    'hypernyms_a': hypernyms_a,
                    'hyponyms_a': hyponyms_a,
                    'meronyms_a': meronyms_a,
                    'holonyms_a': holonyms_a,
               })
        questionid += 1
    solr.add(data)



def getFeatures(tokensTagged, line):
    lemmaS = set()
    hyperS = set()
    hypoS = set()
    meroS = set()
    holoS = set()

    for word, tag in tokensTagged:
        if tag[:2] in nltkWnMap:# and tag != 'NNP':
            sense = lesk(line, word, pos=nltkWnMap.get(tag[:2]))
            # sense = lesk(line, word)
            if not sense:
                continue
            for lem in sense.lemmas():
                lemmaS.add(lem.name())
            for hyper in sense.hypernyms()[:featureMaxLimit]:
                hyperS.add(hyper.name())
            for hypo in sense.hyponyms()[:featureMaxLimit]:
                hypoS.add(hypo.name())
            for mero in sense.part_meronyms()[:featureMaxLimit]:
                meroS.add(mero.name())
            for holo in sense.member_holonyms()[:featureMaxLimit]:
                holoS.add(holo.name())
    return (' '.join(lemmaS), ' '.join(hyperS), ' '.join(hypoS),
            ' '.join(meroS), ' '.join(holoS))


def removestop(tokens):
    removedStop =[]
    for w in tokens:
        if w not in stopWords:
            removedStop.append(w)
    return removedStop

def getLemmas(tokensTagged):
    wnl = WordNetLemmatizer()
    lemmaList = []
    for word, tag in tokensTagged:
        if tag[:2] in nltkWnMap:
            word = wnl.lemmatize(word, pos=nltkWnMap.get(tag[:2]))
        lemmaList.append(word)
    return ' '.join(lemmaList)






def main():
    file = "bitcoin_faq.csv"
    csv_data = read_data(file)
    while(True):
        print("---------------------------")
        print("NLP FAQ MATCHING")
        print("---------------------------")
        print()
        print("1. Task 2:   Bag of words approach")
        print("2. Analysis of task 2")
        print("3. Task 3:   Extract features")
        print("4. Task 4:   Serach using NLP features")
        print("5. Analysis of task 4")
        print("6. Exit")
        sel = 1
        try:
            sel = int(input("Enter your choice: "))
        except (SyntaxError, ValueError):
            print("Enter valid value")
            continue
        if(sel==1):
             task2_default(csv_data)
             # bag_of_words = tokenize(csv_data)
             # question_dict,words = getInputAndTokenize()
             # result = calculatOverlap(question_dict,bag_of_words,words)
             # rank = 1
             # prev = result[0][1]
             # print ("#1"+str(csv_data[result[0][0]]))
             # for i in range(1,10):
             #     if result[i][1] < prev:
             #         rank += 1
             #         prev = result[i][1]
             #     print ("#"+str(rank)+str(csv_data[result[i][0]]))
        elif(sel==2):
             run_analysis(csv_data,2)
        elif(sel==3):
             indexingProcessTask3(csv_data)
        elif(sel==4):
             name = input("Input Question: ").lower()
             question_dict,words = getInputAndTokenize(name)
             searchintask4_default(words)
        elif(sel==5):
             run_analysis(csv_data,5)
        elif(sel==6):
            break

main()
