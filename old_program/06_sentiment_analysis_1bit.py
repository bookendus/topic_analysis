import os

import pandas as pd
from transformers import pipeline

from onebit_utils.tokenization_bitnet import BitnetTokenizer
from onebit_utils.modeling_bitnet import BitnetForCausalLM


IN_DATA_PATH = './data/01_out'
OUT_DATA_PATH = './data/02_out'

SENTIMENTAL_QUESTIONS = {'성분':['긍정적','중립적','부정적'],
                          '맛':['긍정적','중립적','부정적'],
                          '건강':['긍정적','중립적','부정적']}

ko_classifier = pipeline(
    # task='zero-shot-classification',
    task='text-generation',
    device=0,
    model='1bitLLM/bitnet_b1_58-3B'
)


# 파일별로 하나씩 처리한다.
def main(argsß):
    # 입력 파일 로딩 RAW_DATA_PATH에 있는 파일을 DataFrame으로 읽어 드린다.
    input_files = os.listdir(IN_DATA_PATH)

    for in_file in input_files:

        print ("Starting file :%s ..."%in_file)

        df = pd.read_csv(os.path.join(IN_DATA_PATH,in_file), encoding='utf-8')
        print(df.head(3))

        docs = df['generated_text'].tolist()[:30]
        print(docs[:10])

        result = pd.DataFrame()

        for sentence in docs:
            print(sentence)
            for key, candidate_labels in SENTIMENTAL_QUESTIONS.items():

                result_model = ko_classifier(
                    # sequences=sentence,
                    text_inputs=sentence+'%s에 대한 만족감은 {} 이다.'%key,
                    max_length=200,
                    # candidate_labels=candidate_labels,
                    # hypothesis_template='%s에 대한 만족감은 {} 이다.'%key)
                )
                
                # resultDict = {key+"_"+labels:value for labels, value in zip(candidate_labels,result_model["scores"])}
                resultDict = result_model
                print(resultDict)
            


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)