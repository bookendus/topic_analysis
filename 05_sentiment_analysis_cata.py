import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

IN_DATA_PATH = './data/01_out'
OUT_DATA_PATH = './data/05_out'

CANDIDATE_LABELS = ['']

# 입력 파일 로딩 RAW_DATA_PATH에 있는 파일을 DataFrame으로 읽어 드린다.
input_files = os.listdir(IN_DATA_PATH)

# model='cmarkea/bloomz-3b-nli',
# model='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
# model='facebook/bart-large-mnli',
# model='knowledgator/comprehend_it-base', //별로
# model='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
# model='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',

ko_classifier = pipeline(
    task='zero-shot-classification',
    model='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
    device=0,
)


# 파일별로 하나씩 처리한다.

for in_file in input_files:

    print ("Starting file :%s ..."%in_file)

    df = pd.read_csv(os.path.join(IN_DATA_PATH,in_file), encoding='utf-8-sig')
    # print(df.head(3))

    docs = df['generated_text'].tolist()[:500]
    # print(docs[:10])

    result = []

    with tqdm(total=len(docs)) as pbar:
        for sentence in docs:
            returnDict = {'sentence':sentence}
            
            result_model = ko_classifier(
                            sequences=sentence,
                            candidate_labels=CANDIDATE_LABELS,
                            hypothesis_template='구매하는 이유는 {} 이다.')
                
            resultDict = {label:value for label, value in zip(result_model['labels'],result_model["scores"])}
            returnDict.update(resultDict)
            result.append(returnDict)
            pbar.update(1)

    result_df = pd.DataFrame.from_records(result)

    result_df.to_csv(os.path.join(OUT_DATA_PATH,'result_' + in_file), encoding='utf-8-sig')
    
            
