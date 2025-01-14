"""
기업 이름 벡터 스토어 관리 모듈
"""

from typing import List, Tuple
import re
from collections import Counter
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_community.embeddings import ClovaXEmbeddings
from tools.retrieve.financialReport.utils import get_company_names_from_files
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
import numpy as np
import json
from deep_translator import GoogleTranslator
from pykrx import stock
from datetime import datetime, timedelta
import os
import time
import hashlib
import pickle
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

COMPANY_NAME_MAPPING = {
    'posco': '포스코',
    'sk': '에스케이',
    'lg': '엘지',
    'gs': '지에스',
    'kt': '케이티',
    'nh': '엔에이치',
    'cj': '씨제이',
    'ls': '엘에스',
    'db': '디비',
    'ck': '씨케이',
    'spc': '에스피씨',
    'kec': '케이이씨',
    'hl': '에이치엘',
    'mh': '엠에이치',
    'hs': '에이치에스',
    'kc': '케이씨',
    'bnk': '비엔케이',
    'hmm': '에이치엠엠',
    'oci': '오씨아이',
    'dsn': '디에스엔',
    'dsr': '디에스알',
    'kss': '케이에스에스',
    'aks': '에이케이에스',
    'drb': '디알비',
    'stx': '에스티엑스',
    'tyc': '티와이씨',
    'kpx': '케이피엑스',
    'snt': '에스엔티',
    'sgc': '에스지씨',
    'dn': '디엔',
    'tp': '티피',
    'dl': '디엘',
    'e1': '이원',
}

def simple_filter(input_text):
    # 기업명 특별 처리
    input_lower = input_text.lower()
    for eng, kor in COMPANY_NAME_MAPPING.items():
        input_lower = input_lower.replace(eng, kor)
    
    ENGS = ['a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'k',
            'K', 'l', 'L',
            'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T', 'u', 'U', 'v', 'V', 'w',
            'W', 'x', 'X', 'y', 'Y', 'z', 'Z']

    KORS = ['에이', '에이', '비', '비', '씨', '씨', '디', '디', '이', '이', '에프', '에프', '쥐', '쥐', '에이치', '에이치', '아이', '아이', '제이',
            '제이',
            '케이', '케이', '엘', '엘', '엠', '엠', '엔', '엔', '오', '오', '피', '피', '큐', '큐', '알', '알', '에스', '에스', '티', '티', '유',
            '유', '브이', '브이',
            '더블유', '더블유', '엑스', '엑스', '와이', '와이', '지', '지']

    trans = dict(zip(ENGS, KORS)) # 영어와 한글을 대칭하여 딕셔너리화
    is_english = re.compile('[-a-zA-Z]') # 영어 정규화
    temp = is_english.findall(input_lower) # 함수에 들어오는 인자가 영어가 있는경우 temp에 삽입

    result_trans = []
    if len(temp) > 0: # 영어가 temp에 존재하는 경우
        result_trans = ''.join([trans[i] for i in temp]) # 영어를 한글자씩 한글로 매칭
        return result_trans # 매칭된값 리턴
    else:
        return None

BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

DOUBLE_KOREAN_DICT = {
    # 이중자음
    'ㄲ': 'ㄱㄱ',
    'ㄸ': 'ㄷㄷ',
    'ㅃ': 'ㅂㅂ',
    'ㅆ': 'ㅅㅅ',
    'ㅉ': 'ㅈㅈ',
    'ㄳ': 'ㄱㅅ',
    'ㄵ': 'ㄴㅈ',
    'ㄶ': 'ㄴㅎ',
    'ㄺ': 'ㄹㄱ',
    'ㄻ': 'ㄹㅁ',
    'ㄼ': 'ㄹㅂ',
    'ㄽ': 'ㄹㅅ',
    'ㄾ': 'ㄹㅌ',
    'ㄿ': 'ㄹㅍ',
    'ㅀ': 'ㄹㅎ',
    'ㅄ': 'ㅂㅅ',
    # 이중모음
    'ㅐ': 'ㅏㅣ',
    'ㅒ': 'ㅑㅣ',
    'ㅔ': 'ㅓㅣ',
    'ㅖ': 'ㅕㅣ',
    'ㅘ': 'ㅗㅏ',
    'ㅙ': 'ㅗㅐ',
    'ㅚ': 'ㅗㅣ',
    'ㅝ': 'ㅜㅓ',
    'ㅞ': 'ㅜㅔ',
    'ㅟ': 'ㅜㅣ',
    'ㅢ': 'ㅡㅣ'
}

def convert(test_keyword):
    split_keyword_list = list(test_keyword)
    result = list()

    for keyword in split_keyword_list:
        # 한글인지 확인 후 진행합니다.
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - BASE_CODE

            # char1: 초성
            char1 = int(char_code / CHOSUNG)
            result.append(CHOSUNG_LIST[char1])

            # char2: 중성
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            result.append(JUNGSUNG_LIST[char2])

            # char3: 종성
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            if char3==0:
                pass

            else:
                result.append(JONGSUNG_LIST[char3])

        else:
            result.append(keyword)

    return ''.join(result)

def divide_more(s: str):
    """
    '짜장면' 과 '자장면'이 더 가까워질 수 있도록 하는 함수입니다.
    
    'ㅑ' 와 'ㅏ' 는 유사합니다.
    하지만 이것까지는 다루는 것은 상당히 어렵다고 판단해서 제외하였습니다.
    """
    for k, v in DOUBLE_KOREAN_DICT.items():
        s = s.replace(k, v)

    return s

def is_korean(c: str) -> bool:
    """한글 문자인지 확인"""
    return ('ㄱ' <= c <= 'ㅎ') or ('가' <= c <= '힣')

def get_consonant(c: str) -> str:
    """한글 문자의 초성을 반환"""
    if 'ㄱ' <= c <= 'ㅎ':
        return c
    if '가' <= c <= '힣':
        code = ord(c) - ord('가')
        return chr(ord('ㄱ') + code // 588)
    return c

def is_consonant(c: str) -> bool:
    """초성인지 확인"""
    return 'ㄱ' <= c <= 'ㅎ'

def omit_final(c: str) -> str:
    """종성을 제거한 문자 반환"""
    if not ('가' <= c <= '힣'):
        return c
    code = ord(c) - ord('가')
    initial = code // 588
    medial = (code % 588) // 28
    return chr(ord('가') + initial * 588 + medial * 28)

def is_similar(a: str, b: str) -> bool:
    """두 한글 문자의 유사도 확인"""
    if a == b:
        return True
    
    # 둘 중 하나가 초성이면 초성만 비교
    if is_consonant(a) or is_consonant(b):
        return get_consonant(a) == get_consonant(b)
    
    # 그 외에는 종성을 제거한 글자 비교
    return omit_final(a) == omit_final(b)

def calculate_levenshtein_distance(s1: str, s2: str) -> float:
    """개선된 리벤슈타인 거리 계산"""
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
        
    # 행렬 초기화
    matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    
    for i in range(len(s1) + 1):
        matrix[i][0] = i
    for j in range(len(s2) + 1):
        matrix[0][j] = j
        
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            char1 = s1[i-1]
            char2 = s2[j-1]
            
            if char1 == char2:
                cost = 0
            elif is_korean(char1) and is_korean(char2) and is_similar(char1, char2):
                cost = 0.1  # 유사한 한글 문자간 비용
            else:
                cost = 1
                
            matrix[i][j] = min(
                matrix[i-1][j] + 1,      # 삭제
                matrix[i][j-1] + 1,      # 삽입
                matrix[i-1][j-1] + cost  # 대체
            )
    
    return matrix[len(s1)][len(s2)]

class CompanyNameMatcher:
    def __init__(self):
        try:
            self.all_names = list(get_company_names_from_files())
        except FileNotFoundError:
            print("기업 정보 파일을 찾을 수 없습니다.")
            self.all_names = []
        except json.JSONDecodeError:
            print("기업 정보 파일 형식이 잘못되었습니다.") 
            self.all_names = []
        
        self.translator = GoogleTranslator(source='auto', target='en')
        
    def calculate_similarity_score(self, query: str, company_name: str) -> float:
        """개선된 유사도 점수 계산"""
        # 대소문자 구분 없이 비교
        query = query.lower()
        company_name = company_name.lower()
        
        # 기업명 매핑 적용
        for eng, kor in COMPANY_NAME_MAPPING.items():
            query = query.replace(eng, kor)
            company_name = company_name.replace(eng, kor)
        
        # 정확히 일치하는 경우
        if query == company_name:
            return 1.0
            
        # 리벤슈타인 거리 계산
        distance = calculate_levenshtein_distance(query, company_name)
        
        # 거리를 0-1 사이 점수로 정규화
        max_length = max(len(query), len(company_name))
        similarity = 1 - (distance / max_length)
        
        # 초성 매칭 보너스 점수
        if all(is_similar(q, c) for q, c in zip(query, company_name[:len(query)])):
            similarity += 0.1
            
        return min(1.0, similarity)  # 1.0을 넘지 않도록 보정

    def find_matching_company(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """가장 유사한 기업명 찾기"""
        candidates = []
        # 원본 쿼리로 시도
        for official_name in self.all_names:
            similarity = self.calculate_similarity_score(query, official_name)
            candidates.append((official_name, similarity))
        
        # 영문자 한글 변환 시도
        translation = simple_filter(query)
        if translation:
            print(f"영문자 변환 결과: {translation}")
            for official_name in self.all_names:
                similarity = self.calculate_similarity_score(translation, official_name)
                candidates.append((official_name, similarity))
        
        # 중복 제거 및 최고 점수 유지
        best_scores = {}
        for company, score in candidates:
            if company not in best_scores or score > best_scores[company]:
                best_scores[company] = score
        
        # 결과 정렬
        results = [(company, score) for company, score in best_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """점수를 0-1 사이로 정규화"""
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]


class CompanyVectorStore:
    """기업 이름 벡터 스토어 관리 클래스"""
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 초기화 플래그 추가
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, persist_dir: str = "tools/retrieve/financialReport/company_chroma_db",
                 cache_dir: str = "tools/retrieve/financialReport/embedding_cache"):
        """
        Args:
            persist_dir: 벡터 스토어 저장 경로
            cache_dir: 임베딩 결과 캐싱 경로
        """
        # 이미 초기화된 경우 스킵
        if self._initialized:
            return
            
        self.persist_dir = persist_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.embeddings = ClovaXEmbeddings(
            service_app=True,
            model_name="v2",
            timeout=60
        )
        
        # 초기화 완료 표시
        self.vectorstore = None
        self.company_names = get_company_names_from_files()
        self.net_purchase_weights = self._get_net_purchase_weights()
        self._initialized = True

    def _generate_batch_id(self, batch: List[str]) -> str:
        """배치의 고유 ID 생성"""
        content = "".join(batch)
        #print("사용된 내용: ", content)
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _get_cache_path(self, batch_id: str) -> Path:
        """캐시 파일 경로 반환"""
        return self.cache_dir / f"batch_{batch_id}.pkl"

    def _cache_exists(self, batch_id: str) -> bool:
        """캐시 존재 여부 확인"""
        return self._get_cache_path(batch_id).exists()

    def _save_cache(self, batch_id: str, embeddings: List[List[float]]) -> None:
        """임베딩 결과 캐싱"""
        cache_path = self._get_cache_path(batch_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)

    def _load_cache(self, batch_id: str) -> List[List[float]]:
        """캐시된 임베딩 로드"""
        cache_path = self._get_cache_path(batch_id)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=10, max=600)
    )
    def _create_embeddings_with_retry(self, texts: List[str]) -> List[List[float]]:
        """임베딩 생성 (재시도 로직 포함)"""
        return self.embeddings.embed_documents(texts)

    def load_or_create(self, create_flag: bool = False) -> Chroma:
        if self.vectorstore is not None:
            return self.vectorstore
            
        if not create_flag:
            print("기존 기업 벡터스토어를 로드합니다...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            print("기업 벡터스토어를 구축합니다...")
            self.vectorstore = self._create_new_vectorstore()
            
        return self.vectorstore
    
    def _create_new_vectorstore(self) -> Chroma:
        """새로운 기업 벡터스토어 생성"""
        company_names = list(get_company_names_from_files())
        company_names.sort()
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        
        # 배치 처리
        batch_size = 50
        for i in tqdm(range(0, len(company_names), batch_size), desc="기업 이름 처리 중"):
            batch = company_names[i:i + batch_size]
            batch_id = self._generate_batch_id(batch)
            
            # 각 기업의 ID 생성
            ids = [f"company_{i+j}" for j in range(len(batch))]
            
            # 메타데이터 생성
            metadatas = [{"company_name": name} for name in batch]
            
            # 기존 문서 확인
            existing_docs = vectorstore.get(ids=ids)
            existing_ids = set(existing_docs["ids"])
            
            # 새로운 문서만 필터링
            new_batch = []
            new_ids = []
            new_metadatas = []
            for company_name, doc_id, metadata in zip(batch, ids, metadatas):
                if doc_id not in existing_ids:
                    new_batch.append(company_name)
                    new_ids.append(doc_id)
                    new_metadatas.append(metadata)
            
            if not new_batch:  # 새로운 문서가 없으면 건너뛰기
                continue
                
            # 캐시 확인
            cached_embeddings = self._load_cache(batch_id)
            
            if cached_embeddings:
                print(f"캐시된 임베딩을 사용합니다: {batch_id}")
                vectorstore._collection.add(
                    ids=new_ids,
                    embeddings=cached_embeddings[:len(new_batch)],
                    documents=new_batch,
                    metadatas=new_metadatas
                )
            else:
                #print(f"새로운 임베딩을 생성합니다: {batch_id}")
                embeddings = self._create_embeddings_with_retry(new_batch)
                vectorstore._collection.add(
                    ids=new_ids,
                    embeddings=embeddings,
                    documents=new_batch,
                    metadatas=new_metadatas
                )
                self._save_cache(batch_id, embeddings)
            
            # 배치당 2초 대기
            time.sleep(1)
        
        return vectorstore

    def find_similar_companies(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        유사한 기업 이름 검색
        
        Args:
            query: 검색할 기업 이름
            k: 반환할 유사 기업 수
            
        Returns:
            유사한 기업 이름과 정규화된(0~1) 유사도 점수 리스트
            가까운 기업일수록 1에 가깝고, 먼 기업일수록 0에 가까운 점수를 반환
        """
        vectorstore = self.load_or_create()
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        # 유사도 점수 정규화 (0~1)
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        normalized_results = []
        for doc, score in results:
            # 거리가 가까울수록(score가 작을수록) 1에 가깝게, 멀수록(score가 클수록) 0에 가깝게 변환
            normalized_score = ((max_score - score) / score_range) if score_range != 0 else 1.0
            normalized_results.append((doc.metadata["company_name"], normalized_score))
            
        return normalized_results
    
    def _get_net_purchase_weights(self) -> Dict[str, float]:
        """
        개인투자자 순매수 상위 종목 데이터를 가져와서 정규화된 가중치로 변환
        캐시된 데이터가 있고 유효기간이 지나지 않았다면 그것을 사용
        """
        cache_file = "tools/retrieve/financialReport/net_purchase_weights.json"
        
        try:
            # 캐시 파일이 있는지 확인
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # 캐시 데이터의 유효성 검사 (하루 이내)
                cache_date = datetime.strptime(cached_data['date'], "%Y%m%d")
                if datetime.now() - cache_date < timedelta(days=1):
                    print("캐시된 순매수 가중치 사용")
                    return cached_data['weights']
            
            # 새로운 데이터 조회
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=14)).strftime("%Y%m%d")
            
            df = stock.get_market_net_purchases_of_equities(
                start_date, 
                end_date,
                "KOSPI",
                "개인"
            )
            
            # 순매수 금액에 로그 변환 적용
            df['log_amount'] = df['순매수거래량'].apply(
                lambda x: np.sign(x) * np.log1p(abs(x)) if x != 0 else 0
            )
            
            # Min-Max 정규화 적용
            min_val = df['log_amount'].min()
            max_val = df['log_amount'].max()
            df['normalized_weight'] = (df['log_amount'] - min_val) / (max_val - min_val)
            
            # 종목명을 인덱스로 설정하고 딕셔너리로 변환
            df.set_index('종목명', inplace=True)
            weights = df['normalized_weight'].to_dict()
            
            # 캐시 데이터 저장
            cache_data = {
                'date': datetime.now().strftime("%Y%m%d"),
                'weights': weights
            }
            
            # 캐시 디렉토리가 없다면 생성
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            print("새로운 순매수 가중치 계산 및 캐시 저장 완료")
            return weights
            
        except Exception as e:
            print(f"순매수 데이터 조리 실패: {e}")
            # 캐시된 데이터가 있다면 그것이라도 사용
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)['weights']
                except:
                    pass
            return {}

    def hybrid_search(self, query: str, k: int = 5, 
                     matcher_weight: float = 0.5,
                     vector_weight: float = 0.4,
                     purchase_weight: float = 0.1) -> List[Tuple[str, float]]:
        """
        성능이 최적화된 하이브리드 검색
        """
        # 1. 초기 검사 - 영어인지 한글인지 판단
        query = query.strip().upper()
        
        # 2. 약자 처리를 위한 최적화된 매핑
        abbreviation_mapping = {
            'SDI': ['삼성SDI', 'SAMSUNG SDI'],
            'SK': ['SK', '에스케이'],
            'LG': ['LG', '엘지'],
            'GS': ['GS', '지에스'],
        }
        
        # 3. 쿼리 변형 생성 (최소화)
        query_variations = [query]
        
        # 약자가 있는 경우에만 처리
        for abbr, expansions in abbreviation_mapping.items():
            if abbr == query:  # 정확히 일치하는 경우만 처리
                query_variations.extend(expansions)
                break  # 하나 찾으면 중단
        
        # 중복 제거
        query_variations = list(dict.fromkeys(query_variations))
        
        # 4. 매처 검색 최적화
        matcher = CompanyNameMatcher()
        all_results = {}
        
        for query_var in query_variations:
            # 매처 검색
            matcher_results = matcher.find_matching_company(query_var, top_k=k)
            
            # 높은 정확도의 결과가 있으면 벡터 검색 스킵
            high_accuracy_match = False
            
            for company, score in matcher_results:
                if score > 0.9:  # 높은 정확도 임계값
                    high_accuracy_match = True
                    all_results[company] = {
                        'matcher_score': score,
                        'vector_score': 0.0,
                        'found_in_variation': query_var
                    }
            
            # 높은 정확도 매치가 없는 경우에만 벡터 검색 수행
            if not high_accuracy_match:
                vector_results = self.find_similar_companies(query_var, k=k)
                
                for company, score in vector_results:
                    if company not in all_results:
                        all_results[company] = {
                            'matcher_score': 0.0,
                            'vector_score': score,
                            'found_in_variation': query_var
                        }
                    else:
                        all_results[company]['vector_score'] = max(
                            all_results[company]['vector_score'], 
                            score
                        )
        
        # 5. 최종 점수 계산 (최적화)
        final_scores = {}
        
        for company, scores in all_results.items():
            # 기본 점수 계산
            base_score = (scores['matcher_score'] * matcher_weight + 
                        scores['vector_score'] * vector_weight)
            
            # 약자 매칭에 대한 보너스 점수 (단순화)
            if query in abbreviation_mapping and \
            any(exp in scores['found_in_variation'] for exp in abbreviation_mapping[query]):
                base_score *= 1.2
            
            final_scores[company] = base_score
        
        # 6. 상위 k개 결과만 반환
        final_results = sorted(
            [(company, score) for company, score in final_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return final_results