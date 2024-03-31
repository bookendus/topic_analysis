# 여러 문장으로 구성된 리뷰는 한문장에 한 줄씩 나눈다.

import os, re
import pandas as pd
from tqdm import tqdm

# 디렉토리를 정해준다.
RAW_DATA_PATH = "./data/01_out"
OUT_DATA_PATH = "./data/02_out"

# 입력 파일 로딩 RAW_DATA_PATH에 있는 파일을 DataFrame으로 읽어 드린다.
input_files = os.listdir(RAW_DATA_PATH)

def split_text(text):
    
    out_list = []
    #맨앞과 맨투의 공백을 제거한다.
    text = text.strip()
    line = ''
    deli = ['.','?']
    for char in text:
        line += char
        if char in deli:
            out_list.append(line.strip())
            line = ''
    return out_list


# 파일별로 하나씩 처리한다.
for in_file in input_files:

    outList = []
    df = pd.read_csv(os.path.join(RAW_DATA_PATH,in_file), encoding='utf-8')
    print ("Starting file :%s ..."%in_file)
    for text in tqdm(df['generated_text']):
        outList.extend(split_text(text))
    out_df = pd.DataFrame({'generated_text' : outList})
    out_df['word_count'] = out_df['generated_text'].apply(lambda x: len(x.split(' ')))
    out_df.to_csv(os.path.join(OUT_DATA_PATH,in_file))

    
    
