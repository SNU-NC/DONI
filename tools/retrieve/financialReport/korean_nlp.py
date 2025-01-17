from kiwipiepy import Kiwi
from konlpy.tag import Mecab
import math
from typing import List, Dict, Tuple, Optional
from tools.retrieve.financialReport.utils import get_company_names_from_files
from tools.retrieve.financialReport.company_name_vector_store import CompanyVectorStore

digit_name = ['영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
unit = ['', '십', '백', '천']
unit_10k = ['', '만', '억', '조', '경', '해', '자', '양', '구', '간', '정', '재', '극', '항하사', '아승기', '나유타', '불가사의', '무량대수']

numbers = [
    # Digits
    ('1', 1),
    ('2', 2),
    ('3', 3),
    ('4', 4),
    ('5', 5),
    ('6', 6),
    ('7', 7),
    ('8', 8),
    ('9', 9),

    ("일", 1),
    ("이", 2),
    ("삼", 3),
    ("사", 4),
    ("오", 5),
    ("육", 6),
    ("칠", 7),
    ("팔", 8),
    ("구", 9),

    ("하나", 1),
    ("한", 1),
    ("두", 2),
    ("둘", 2),
    ("세", 3),
    ("셋", 3),
    ("네", 4),
    ("넷", 4),
    ("다섯", 5),
    ("여섯", 6),
    ("일곱", 7),
    ("여덟", 8),
    ("여덜", 8),
    ("아홉", 9),

    # Digits + Unit
    ("스물", 20),
    ("서른", 30),
    ("마흔", 40),
    ("쉰",   50),
    ("예순", 60),
    ("일흔", 70),
    ("여든", 80),
    ("아흔", 90),

    # Mini Unit
    ("열", 10),
    ("십", 10),
    ("백", 10**2),
    ("천", 10**3),

    # Unit
    ("만", 10**4),
    ("억", 10**8),
    ("조", 10**12),
    ("경", 10**16),
    ("해", 10**20),
]

float_nums = [
    ("일", 1),
    ("이", 2),
    ("삼", 3),
    ("사", 4),
    ("오", 5),
    ("육", 6),
    ("칠", 7),
    ("팔", 8),
    ("구", 9)
]
class KoreanTextAnalyzer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 초기화 플래그 추가
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # 이미 초기화된 경우 스킵
        if self._initialized:
            return
            
        # 실제 초기화 작업
        self.kiwi = Kiwi(model_type="sbg")
        self.kiwi.space_tolerance = 2
        self.mecab = Mecab()
        self.company_names = set()
        self.company_name_db = CompanyVectorStore()
        
        # 초기화 완료 표시
        self._initialized = True
    
    def get_company_names(self) -> set: # 사전에 추가로 등록된 회사명 확인
        return self.company_names
    
    def add_company(self, name: str) -> None:
        """기업명을 사전에 등록"""
        # 기본 형태 등록 (공백 없는 버전)
        self.kiwi.add_user_word(name, 'NNP', score=100)
        self.company_names.add(name)
    
    def add_companies_from_list(self, companies: List[str] = None) -> None:
        """기업명 리스트를 일괄 등록"""
        if companies is None:
            companies = list(get_company_names_from_files())
        for company in companies:
            self.add_company(company)

    def kr2num(self, kr_str, ignore_units=False):
        if len(kr_str) >= 2:
            ignore_units = False
        decode_result = []
        result = 0
        temp_result = 0
        index = 0

        float_dividing = kr_str.split("점")
        float_result = ""
        if len(float_dividing) == 2:
            kr_str = float_dividing[0]
            float_num = float_dividing[1]
            for c in float_num:
                for float_num, float_value in float_nums:
                    if c == float_num:
                        float_result += str(float_value)
                        break
            if len(float_result) == 0:
                float_result = 0.0
            else:
                float_result = float("0." + float_result)
        else:
            float_result = 0.0

        while index < len(kr_str):
            for number, true_value in numbers:
                if index + len(number) <= len(kr_str):
                    if kr_str[index:index + len(number)] == number:
                        decode_result.append((true_value, math.log10(true_value).is_integer()))
                        if len(number) == 2:
                            index += 1
                        break
            index += 1
        #print(decode_result)

        for index, (number, is_natural) in enumerate(decode_result):
            if is_natural:
                if not ignore_units and math.log10(number) >= 4 and (math.log10(number) % 4 == 0):
                    result += temp_result * number
                    temp_result = 0
                elif index - 1 >= 0:
                    if not decode_result[index - 1][1]:
                        temp_result += number * decode_result[index - 1][0]
                    else:
                        if index == 1 and temp_result == 1:
                            temp_result = number
                        else:
                            temp_result += number
                else:
                    temp_result += number
                #print("temp_result", temp_result)

            else:
                if index + 1 == len(decode_result):
                    temp_result += number
                elif not decode_result[index + 1][1]:
                    temp_result += number
                elif not ignore_units and math.log10(decode_result[index + 1][0]) > 3 and (math.log10(decode_result[index + 1][0]) - 4) % 4 == 0:
                    temp_result += number

        result += temp_result

        if float_result != 0.0:
            result += float_result

        return result

    def split_digit(self, num:int, div:int = 10) -> list:
        ret = []
        while num!=0:
            num, rem = divmod(num, div)
            ret.append(rem)
        return ret

    def num2kr(self, num : int, mode=1) -> str:
        if num>=pow(10000, len(unit_10k)+1):
            raise ValueError("Value exceeds 10e72; cannot be read")

        digit_10k = self.split_digit(num, 10000)

        if mode==1:
            for i in range(len(digit_10k)):
                digit = self.split_digit(digit_10k[i])
                tmp = []
                for j in range(len(digit)):
                    if digit[j]!=0:
                        tmp.append(digit_name[digit[j]] + unit[j])
                digit_10k[i] = ''.join(reversed(tmp))

        kr_str = []
        for i in range(len(digit_10k)):
            if digit_10k[i]!=0:
                kr_str.append(str(digit_10k[i]) + unit_10k[i])

        glue = '' if mode==1 else ' '
        kr_str = glue.join(reversed(kr_str))

        return kr_str

    def process_number_tokens(self, tokens):
        merged = []
        nr_group = []
        
        for token, pos in tokens:
            if pos == 'NR':
                nr_group.append(token.strip())
            else:
                if nr_group:  # NR 그룹이 있으면 병합
                    merged_num = ''.join(nr_group)
                    merged.append((merged_num, 'NR'))
                    nr_group = []
                merged.append((token.strip(), pos))
        
        # 마지막 NR 그룹 처리
        if nr_group:
            merged_num = ''.join(nr_group)
            merged.append((merged_num, 'NR'))
        
        return merged

    def normalize_number(self, text: str) -> Tuple[str, List[Dict]]:
        """숫자 표현을 정규화"""
        def roman_to_int(roman):
            roman_values = {'Ⅰ': 1, 'Ⅱ': 2, 'Ⅲ': 3, 'Ⅳ': 4, 'Ⅴ': 5}
            return roman_values.get(roman, roman)

        # 쉼표 제거 및 연속된 SN 토큰 병합
        processed_tokens = []
        temp_sn = []
        
        for token, pos in self.mecab.pos(text):
            if token == ',' and pos == 'SC':
                continue
                
            if pos == 'SN':
                temp_sn.append(token)
            else:
                if temp_sn:
                    merged_num = ''.join(temp_sn)
                    processed_tokens.append((merged_num, 'SN'))
                    temp_sn = []
                processed_tokens.append((token, pos))
                
        if temp_sn:
            merged_num = ''.join(temp_sn)
            processed_tokens.append((merged_num, 'SN'))

        # NR과 SN 토큰 처리
        tmp_tokens = []
        i = 0
        while i < len(processed_tokens):
            token, pos = processed_tokens[i]
            if pos == 'SN' and i + 1 < len(processed_tokens) and processed_tokens[i + 1][1] == 'NR':
                next_token, next_pos = processed_tokens[i + 1]
                if (i == 0 or processed_tokens[i - 1][1] not in ['NR', 'SN']) and (i + 2 >= len(processed_tokens) or processed_tokens[i + 2][1] not in ['NR', 'SN']):
                    tmp_tokens.append((str(int(token) * self.kr2num(next_token, ignore_units=True)), 'SN'))
                    i += 1
                else:
                    tmp_tokens.append((token, pos))
            else:
                tmp_tokens.append((token, pos))
            i += 1

        # SN과 NR 토큰 처리
        final_tokens = []
        for token, pos in tmp_tokens:
            if pos == 'SN':
                try:
                    if token in ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ']:
                        number = roman_to_int(token)
                    else:
                        number = int(token)
                    final_tokens.append((self.num2kr(number), 'NR'))
                except ValueError:
                    final_tokens.append((token, 'NR'))
            else:
                final_tokens.append((token, pos))

        # 남아있는 NR 토큰 합치기
        final_tokens = self.process_number_tokens(final_tokens)
        final_strings = []
        # 남아있는 NR -> 숫자로 변환
        for i, (token, pos) in enumerate(final_tokens):
            if pos == 'NR':
                normalized_num = str(self.kr2num(token))
                final_strings.append(normalized_num)
            else:
                final_strings.append(token)
        # 결과 문장 재구성
        return self.kiwi.space(" ".join(final_strings))

    def company_name_rewriter(self, text: str) -> str:
        """텍스트에서 기업명 찾아서 {기업명} 기업으로 변환"""
        text = self.mecab.nouns(text)
        return text

    def normalize_text(self, text: str) -> str:
        """텍스트 -> 한글 단위를 모두 숫자로 변경한 문자열"""
        return self.normalize_number(text)
    
    def get_top_related_companies(self, text: str) -> str:
        """텍스트에서 유사한 기업명 1개 찾기"""
        return self.company_name_db.find_similar_companies(text, k=1)[0]