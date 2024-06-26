import torch, os, kss
import pandas as pd

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from datasets import Dataset

from tqdm.auto import tqdm

# 디렉토리를 정해준다.
RAW_DATA_PATH = "./data/01_in"
OUT_DATA_PATH = "./data/01_out"

# 후보정 문자 
REPLACING_TEXT = {'gm':'gmo', '중 위험':'중위험', '저 위험':'저위험', '고 위험':'고위험', '?.':'?' }

# 리뷰는 구어체가 많아 구어체 오타수정 언어모델 사용
model = T5ForConditionalGeneration.from_pretrained('j5ng/et5-typos-corrector')
tokenizer = T5Tokenizer.from_pretrained('j5ng/et5-typos-corrector', legacy=False)

# 파이프라인 정의
typos_corrector = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    framework="pt",
)

# Prompt를 넣어준다.
def add_prefix(sentence):
    sentence["talk_ko"] = sentence['talk_ko'].replace(".","")
    sentence["talk_ko"] = "맞춤법 고쳐주세요: " + sentence["talk_ko"]
    return sentence

# 마지막에 .이나 ?가 없으면 .을 붙인다.
def add_dot(text):
    if text[-1:] in [ '.', '?'] :
        text = text + ' '
    else:
        text = text + '. '
    return text



# kss를 사용하여 문장 분리 및 구두점 추가
def split_sentences(text):
    sentences = kss.split_sentences(text)
    
    return ''.join(list(map(add_dot,sentences)))


# 입력 파일 로딩 RAW_DATA_PATH에 있는 파일을 DataFrame으로 읽어 드린다.
input_files = os.listdir(RAW_DATA_PATH)

# 파일별로 하나씩 처리한다.
for in_file in input_files:

    df = pd.read_csv(os.path.join(RAW_DATA_PATH,in_file), encoding='utf-8')

    print ("Starting file :%s ..."%in_file)

    dataset = Dataset.from_pandas(df).map(add_prefix)
    
    out_json = tqdm(typos_corrector(dataset["talk_ko"], max_length=128, num_beams=5, early_stopping=False),desc="Correcting")
    out_df_b = pd.DataFrame(out_json)

    #후처리 단어 수정
    for before, after in REPLACING_TEXT.items():
        out_df_b['generated_text'] = out_df_b['generated_text'].str.replace(before, after)

    # 'talk_ko_ch' 열에 대해 함수 적용
    out_df_b['generated_text'] = out_df_b['generated_text'].apply(split_sentences)


    out_df =  pd.concat([df,out_df_b],axis=1) 

    out_df.to_csv(os.path.join(OUT_DATA_PATH,in_file))


print(" Completed %d files"%len(input_files))

