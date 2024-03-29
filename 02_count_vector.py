import os

import pandas as pd
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import CountVectorizer

IN_DATA_PATH = './data/01_out'
OUT_DATA_PATH = './data/02_out'

# 입력 파일 로딩 RAW_DATA_PATH에 있는 파일을 DataFrame으로 읽어 드린다.
input_files = os.listdir(IN_DATA_PATH)

# 불용어를 정의한다
user_stop_word = [  ]

# 토크나이저에 ***만 추가한다      https://incredible.ai/nlp/2016/12/28/NLP/
extract_pos_list = ["NNG", "NNP", "NNB", "NR", "NP", "VV", "VA","MAG","IC"]

class CustomTokenizer:
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


custom_tokenizer = CustomTokenizer(kiwi)
vectorizer = CountVectorizer(tokenizer=custom_tokenizer)


# 파일별로 하나씩 처리한다.

for in_file in input_files:

    print ("Starting file :%s ..."%in_file)

    df = pd.read_csv(os.path.join(IN_DATA_PATH,in_file), encoding='utf-8-sig')
    # print(df.head(3))

    docs = df['generated_text'].tolist()[:500]
    # print(docs[:10])

    topics = vectorizer.fit_transform(docs)
    
    column_names = vectorizer.get_feature_names_out()
    df = pd.DataFrame(topics.toarray(), columns = column_names)

    df_out = pd.DataFrame([ (col, df[col].sum()) for col in df.columns ], columns = ['word','count'])
    
    df_out.to_csv(os.path.join(OUT_DATA_PATH,in_file), encoding='utf-8-sig')

    # print(df_out)
    # df = pd.DataFrame([ (k ,v) for k, v in topics.vocabulary_.items()], columns=['word','count'])
    # df.to_csv(os.path.join(OUT_DATA_PATH,in_file), encoding='utf-8-sig')

