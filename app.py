import boto3  # The AWS SDK for Python
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

import config
import os

# Declare a constant variable
TARGET_LANGUAGE_CODE = 'en'
SOURCE_LANGUAGE_CODE = 'ko'

# Similarity Criterion Percent
SIMILARITY_CRITERION_PERCENT = 10


def get_mongo_client():
    username = config.MONGODB_USERNAME
    password = config.MONGODB_PASSWORD
    host = config.MONGODB_HOST
    port = config.MONGODB_PORT

    # Create a MongoDB connection URI
    mongo_uri = f'mongodb://{username}:{password}@{host}:{port}/'

    # Create the MongoDB client and return it
    return MongoClient(mongo_uri)


def cosine_similarity_to_percent_general(cosine_similarity):
    normalized_value = (cosine_similarity + 1) / 2
    return normalized_value * 100


def lambda_handler(event, context):
    os.environ['TRANSFORMERS_CACHE'] = "/tmp"

    # Configure AWS Translate client
    translate = boto3.client(service_name='translate',
                             aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                             region_name=config.AWS_SEOUL_REGION)

    mentor_nickname = event['mentor_nickname']
    mentee_nickname = event['mentee_nickname']
    question_origin = event['question_origin']
    question_summary = event['question_summary']
    question_time = datetime.now()  # 데이터를 받은 시간을, 질문이 입력된 시간으로 설정

    """ 받아온 데이터 중, 세 줄 요약된 질문을 AWS Translate API를 통해 영어로 번역 """
    translation_response = translate.translate_text(Text=question_summary, SourceLanguageCode=SOURCE_LANGUAGE_CODE,
                                                    TargetLanguageCode=TARGET_LANGUAGE_CODE)

    """ Extract the translated text from the response """
    translated_summary_text_en = translation_response['TranslatedText']

    """ Connect MongoDB """
    mongo_client = get_mongo_client()
    menjil_db = mongo_client['menjil']
    qa_list_collection = menjil_db['qa_list']

    """qa_list collection에 접근해서, Spring Boot에서 받아온 정보(멘토 닉네임, 멘티 닉네임, 원본 질문, 세 줄 요약된 질문)와 영어 번역본을 먼저 저장"""
    document = {
        # 마지막에 붙는 '\n' 제거
        'mentee_nickname': mentee_nickname,
        'mentor_nickname': mentor_nickname,
        'question_origin': question_origin[:-1] if question_origin.endswith('\n') else question_origin,
        'question_summary': question_summary[:-1] if question_summary.endswith('\n') else question_summary,
        'question_summary_en': translated_summary_text_en[:-1]
        if translated_summary_text_en.endswith('\n') else translated_summary_text_en,
        'question_time': question_time,
        'answer': None,
        'answer_time': None
    }
    insert = qa_list_collection.insert_one(document)  # save a document

    """ 멘토가 답변한 내역이 있는 문답 데이터를 모두 불러온다 """
    filter_ = {
        'mentor_nickname': mentor_nickname,
        'answer': {'$exists': True, '$ne': None}
    }
    projection_ = {
        'mentee_nickname': False,
        'mentor_nickname': False,
        'question_origin': False
    }
    # Retrieve the documents and store them in the data(list)
    data = list(qa_list_collection.find(filter_, projection_))
    print('data: ', data)

    """ 문장 유사도 검증 """
    """ 1. 유사도 검사"""
    question_summary_en_list = [doc['question_summary_en'] for doc in data]
    # for idx, qe in enumerate(question_summary_en_list):
    #     print(f'질문{idx + 1}: {qe}')

    """ 1-1. 기존에 데이터가 3개 미만으로 존재할 경우, 아래 과정을 거치지 않고 바로 빈 리스트 리턴"""
    if len(question_summary_en_list) < 3:
        return []

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/tmp')
    query_embedding = model.encode(translated_summary_text_en, convert_to_tensor=True)
    passage_embedding = model.encode(question_summary_en_list, convert_to_tensor=True)

    # Use Cosine Similarity
    cos_score = util.cos_sim(query_embedding, passage_embedding)
    # Normalize
    cos_score_percent = cosine_similarity_to_percent_general(cos_score)
    cos_score_percent_list = cos_score_percent.tolist()[0]

    """ 2. 계산된 데이터 중 유사도 상위 3개 데이터 추출 """
    similarity_list = [{'similarity_percent': 0}, {'similarity_percent': 0}, {'similarity_percent': 0}]
    for doc, score in zip(data, cos_score_percent_list):
        doc['similarity_percent'] = score
        sim_list = [d['similarity_percent'] for d in similarity_list]
        if score > min(sim_list):
            idx_min = sim_list.index(min(sim_list))
            similarity_list[idx_min] = doc

    """ 3. 유사도 점수가 기준 점수(SIMILARITY_CRITERION_POINT) 이하인 데이터 삭제 """
    # result_similarity_list = []
    # for doc in similarity_list:
    #     if doc['similarity_percent'] > SIMILARITY_CRITERION_PERCENT:
    #         result_similarity_list.append(doc)

    # 유사도 상위 3개의 데이터 출력
    # print(result_similarity_list)

    """ 결과가 3개 미만일 경우, 빈 리스트를 Spring Boot로 리턴"""
    if len(similarity_list) < 3:
        return []

    """ 요약된 질문과 답변을 DTO로 담아서 리턴(Spring Boot로 전달) """
    # List of DTOs
    data_list = []
    for i in similarity_list:
        dict_ = dict()
        dict_['question_summary'] = i.get('question_summary')
        dict_['answer'] = i.get('answer')
        dict_['similarity_percent'] = round(i.get('similarity_percent'), 2)  # Rounded to 2 decimal places
        data_list.append(dict_)
        print(dict_)

    # Sort the data_list by 'similarity_percent' in descending order
    data_list = sorted(data_list, key=lambda x: x['similarity_percent'], reverse=True)

    return data_list
