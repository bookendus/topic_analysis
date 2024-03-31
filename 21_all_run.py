import pandas as pd
import os

# soynlp tokenizer
from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor
from soynlp.word import WordExtractor

# kiwi tokenizer
from kiwipiepy import Kiwi

# mecab tokenizer
from mecab import MeCab

from sklearn.feature_extraction.text import CountVectorizer

from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from transformers import pipeline

#bertopic lib
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

########## section 0. 상수
CANDIDATE_LABELS = ['첨가물','성분','위험','아기','안심','칼로리','중위험','아이']



########## section 1. read files ###########
IN_DATA_PATH = './data/01_out'
OUT_DATA_PATH = './data/21_out'

in_data_files = []
in_docs = {}
total_docs = []

# 폴더내 파일리스트를 가져온다.
input_files = os.listdir(IN_DATA_PATH)

# 입력 파일 로딩 RAW_DATA_PATH에 있는 파일을 DataFrame으로 읽어 드린다.
for in_file in input_files:
    print ("Loading file :%s ..."%in_file)
    df = pd.read_csv(os.path.join(IN_DATA_PATH,in_file), encoding='utf-8-sig')
    docs = df['generated_text'].tolist()
    in_data_files.append(df)
    in_docs[in_file] =  docs
    total_docs.extend(docs)    


########## section 2-1. soynlp ###########
    
# 불용어를 정의한다
user_stop_word = [ "거", "바", "뻥", "중", "눌", ]

# 학습한다.
noun_extractor = LRNounExtractor()
nouns = noun_extractor.train_extract(total_docs)

word_extractor = WordExtractor(
    min_frequency=4, # example
    min_cohesion_forward=0.05,
    min_right_branching_entropy=0.0
)

word_extractor.train(total_docs)
words = word_extractor.extract()

cohesion_score = {word:score.cohesion_forward for word, score in words.items()}

noun_scores = {noun:score.score for noun, score in nouns.items()}
combined_scores = {noun:score + cohesion_score.get(noun, 0)
    for noun, score in noun_scores.items()}
combined_scores.update(
    {subword:cohesion for subword, cohesion in cohesion_score.items()
    if not (subword in combined_scores)}
)

class soynlp_CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, text):
        result = list()
        for word in self.tagger.tokenize(text):
            # 명사이고, 길이가 2이상인 단어이고, 불용어 리스트에 없으면 추가하기
            if word not in user_stop_word: #len(word[0]) > 1 and
                result.append(word)
        return result

soynlp_tokenizer = LTokenizer(scores=combined_scores)
soynlp_custom_tokenizer = soynlp_CustomTokenizer(soynlp_tokenizer)
soynlp_vectorizer = CountVectorizer(tokenizer=soynlp_custom_tokenizer)

############ section 2-2 kiwi ##############

# 불용어를 정의한다
user_stop_word = [ "거", "바", "뻥", "눌", ]

# 토크나이저에 ***만 추가한다      https://incredible.ai/nlp/2016/12/28/NLP/
extract_pos_list = ["NNG", "NNP", "NNB", "NR", "NP", "VV", "VA","MAG","IC"]

class kiwi_CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, text):
        result = list()
        for word in self.tagger.tokenize(text):
            # 명사이고, 길이가 2이상인 단어이고, 불용어 리스트에 없으면 추가하기
            if word[1] in extract_pos_list and word[0] not in user_stop_word: #len(word[0]) > 1 and
                result.append(word[0])
        return result

# 커스텀 단어 등록
kiwi = Kiwi()
kiwi.add_user_word("노브랜드","NNP")
kiwi.add_user_word("한살림","NNP")
kiwi.add_user_word("이마트","NNP")
kiwi.add_user_word("피코크","NNP")
kiwi.add_user_word("누네띠네","NNP")
kiwi.add_user_word("동원참치","NNP")
kiwi.add_user_word("떡뻥","NNP")
kiwi.add_user_word("베지밀","NNP")
kiwi.add_user_word("네스카페","NNP")
kiwi.add_user_word("맥스봉","NNP")
kiwi.add_user_word("오뚜기","NNP")
kiwi.add_user_word("오징어집","NNP")
kiwi.add_user_word("오곡쿠키","NNP")

kiwi.add_user_word("10대","NR")
kiwi.add_user_word("20대","NR")
kiwi.add_user_word("30대","NR")
kiwi.add_user_word("40대","NR")
kiwi.add_user_word("50대","NR")
kiwi.add_user_word("베스킨라빈스","NNP")
kiwi.add_user_word("이산화규소","NNP")
kiwi.add_user_word("중위험","NNP")
kiwi.add_user_word("고위험","NNP")
kiwi.add_user_word("저위험","NNP")
kiwi.add_user_word("우리밀","NNP")
kiwi.add_user_word("열일","NNG")
kiwi.add_user_word("유전자변형","NNG")
kiwi.add_user_word("유화증진제","NNG")
kiwi.add_user_word("유화제","NNG")
kiwi.add_user_word("스프레드","NNG")
kiwi.add_user_word("식품첨가물","NNG")

kiwi_custom_tokenizer = kiwi_CustomTokenizer(kiwi)
kiwi_vectorizer = CountVectorizer(tokenizer=kiwi_custom_tokenizer)


###############  section 2-3 mecab ###############
class mecab_CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        word_tokens = self.tagger.morphs(sent)
        result = [word for word in word_tokens]
        return result


mecab_custom_tokenizer = mecab_CustomTokenizer(MeCab())
mecab_vectorizer = CountVectorizer(tokenizer=mecab_custom_tokenizer)


############ section 2-9 make list of vectorizer ###########
vectorizers = {'soynlp': soynlp_vectorizer, 'kiwi': kiwi_vectorizer, 'mecab': mecab_vectorizer}


############ section 3 ################

# Pre-calculate embeddings
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
embeddings = embedding_model.encode(total_docs, show_progress_bar=True)

# 차원축소
hdbscan_model = HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# zero-shot
ko_classifier = pipeline(
    task='zero-shot-classification',
    model='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
    device=0,
    # hypothesis_template='구매하는 이유는 {} 이다.',
)


######### 각 Vectorizer 별 실행 ###########

for vectoerizer_name, vectorizer in vectorizers.items():

    print("Vectorizer: =========> " , vectoerizer_name, " <==================")

    for docs_name, docs in in_docs.items():

        print("Document: ", docs_name, " <==============")
        topic_model = BERTopic(
                embedding_model=embedding_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer,
                nr_topics="auto", # 문서를 대표하는 토픽의 갯수
                # top_n_words=4,
                zeroshot_topic_list=CANDIDATE_LABELS,
                zeroshot_min_similarity=.5,
                # representation_model=ko_classifier,
                representation_model=KeyBERTInspired(),
                calculate_probabilities=True
        )
            
        topics, probs = topic_model.fit_transform(docs)
        topic_model.get_topic_info().to_csv(os.path.join(OUT_DATA_PATH,vectoerizer_name+"_"+docs_name+"_get_topic_info.csv" ))