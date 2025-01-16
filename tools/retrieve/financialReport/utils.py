import logging
import os
from typing import Set,  List
from typing import Any, Optional
import json
import pandas as pd
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.documents import Document
import FinanceDataReader as fdr

def create_document_merge_chain():
    def merge_and_restructure_documents(docs: List[Document]) -> List[Document]:
        merged_docs = []
        current_text = ""
        current_table = ""
        current_metadata = {}
        MAX_DOC_LENGTH = 2000

        for doc in docs:
            # 테이블과 텍스트 분리
            table_content = ""
            if 'table' in doc.metadata:
                table_content = format_table(doc.metadata['table'])
            
            # 현재 문서의 내용 구성
            new_text = doc.page_content.strip() if doc.page_content else ""
            new_table = table_content if table_content else ""
            
            # 병합될 전체 내용 미리보기
            merged_text = ""
            if current_text or new_text:
                merged_text = "관련 텍스트:\n" + \
                    ((current_text + "\n" + new_text) if current_text and new_text 
                     else (current_text or new_text))
            
            merged_table = ""
            if current_table or new_table:
                merged_table = "관련 테이블:\n" + \
                    ((current_table + "\n" + new_table) if current_table and new_table 
                     else (current_table or new_table))
            
            combined_content = "\n\n".join(filter(None, [merged_text, merged_table]))

            # 길이 체크 및 문서 생성
            if len(combined_content) > MAX_DOC_LENGTH:
                if current_text or current_table:
                    # 현재까지 누적된 내용으로 문서 생성
                    current_content = ""
                    if current_text:
                        current_content += "관련 텍스트:\n" + current_text
                    if current_table:
                        if current_content:
                            current_content += "\n\n"
                        current_content += "관련 테이블:\n" + current_table
                    
                    merged_docs.append(Document(
                        page_content=current_content,
                        metadata=current_metadata.copy()
                    ))
                
                # 새로운 내용으로 초기화
                current_text = new_text
                current_table = new_table
                current_metadata = {k:v for k,v in doc.metadata.items() if k != 'table'}
            else:
                # 내용 누적
                current_text = (current_text + "\n" + new_text) if current_text and new_text else (current_text or new_text)
                current_table = (current_table + "\n" + new_table) if current_table and new_table else (current_table or new_table)
                current_metadata.update({k:v for k,v in doc.metadata.items() if k != 'table'})

        # 마지막 남은 내용 처리
        if current_text or current_table:
            final_content = ""
            if current_text:
                final_content += "관련 텍스트:\n" + current_text
            if current_table:
                if final_content:
                    final_content += "\n\n"
                final_content += "관련 테이블:\n" + current_table
                
            merged_docs.append(Document(
                page_content=final_content,
                metadata=current_metadata
            ))

        return merged_docs

    return merge_and_restructure_documents

def get_company_names_from_files(data_dir: str = "data/") -> Set[str]:
    """
    data/all_company_names.json 파일에서 기업명 목록을 불러옴
    
    Args:
        data_dir: 데이터 폴더 경로 (기본값: "data/")
        
    Returns:
        기업명 집합 (Set)
    """
    # 현재 작업 디렉토리 기준으로 json 파일 경로 생성

    # ./data/all_company_names.json 을 찾아오는 것임 
    json_path = os.path.join(os.getcwd(), data_dir, "all_company_names.json")
    
    # 파일이 존재하지 않으면 빈 집합 반환
    if not os.path.exists(json_path):
        print(f"파일이 존재하지 않습니다: {json_path}")
        return set()
    logging.info(f"파일이 존재합니다: {json_path}")
    # JSON 파일에서 기업명 목록 불러오기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            company_names = set(data.get('companies', []))
        return company_names
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return set()

def format_table(table_data: Any) -> str:
    """테이블 데이터를 보기 좋은 형식으로 변환"""
    if not table_data:
        return table_data
            
    try:
        # JSON 문자열인 경우 처리
        if isinstance(table_data, str):
            if table_data.startswith('"table":'):
                table_data = table_data.replace('"table":', '').strip()
            try:
                # JSON 파싱 시도
                table_data = json.loads(table_data)
            except json.JSONDecodeError:
                # 기존 파싱 로직 유지
                cleaned = table_data.strip('[]')
                rows = [row.strip().strip('[]').split(',') for row in cleaned.split('],[')]
                table_data = [[item.strip().strip("'\"") for item in row] for row in rows]
        
        if isinstance(table_data, list) and len(table_data) > 0:
            # 첫 번째 행이 문자열인 경우 분리
            if isinstance(table_data[0], str):
                table_data = [row.strip('[]').split(',') for row in table_data]
                table_data = [[item.strip().strip("'\"") for item in row] for row in table_data]
            
            # DataFrame 생성 및 문자열로 변환
            df = pd.DataFrame(table_data)
            if len(df) > 0:
                # 첫 번째 행을 헤더로 사용
                df.columns = df.iloc[0]
                df = df.iloc[1:]
                # TH 접두사 제거
                df.columns = [str(col).replace('TH', '') for col in df.columns]
            return df.to_string(index=False)
    except Exception as e:
        print(f"Error: {e}")  # 디버깅용
        return str(table_data)
    
    return str(table_data)


# def format_table(table_data: Any) -> str:
#     if not table_data:
#         return table_data
            
#     try:
#         if isinstance(table_data, list) and len(table_data) > 0:
#             # 모든 행이 리스트인 경우 (2차원 테이블)
#             if all(isinstance(row, list) for row in table_data):
#                 # 각 열의 최대 너비 계산
#                 col_widths = []
#                 for col in zip(*table_data):
#                     col_widths.append(max(len(str(cell)) for cell in col))
                
#                 # 포맷팅된 행 생성
#                 formatted_rows = []
#                 for row in table_data:
#                     formatted_cells = [
#                         str(cell).ljust(width) 
#                         for cell, width in zip(row, col_widths)
#                     ]
#                     formatted_rows.append(" ".join(formatted_cells))
                
#                 return "\n".join(formatted_rows)
#             else:
#                 # 1차원 리스트인 경우
#                 max_key_width = max(len(str(table_data[i])) for i in range(0, len(table_data), 2))
#                 formatted_rows = []
#                 for i in range(0, len(table_data), 2):
#                     if i + 1 < len(table_data):
#                         key = str(table_data[i]).ljust(max_key_width)
#                         value = str(table_data[i+1])
#                         formatted_rows.append(f"{key} | {value}")
#                 return "\n".join(formatted_rows)
                
#     except Exception as e:
#         print(f"Error in format_table: {e}")
#         return str(table_data)
    
#     return str(table_data)

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


def pretty_print_docs(documents: List[Document]) -> None:
    """검색 종료 시 호출"""
    print("\n" + "="*80)
    print(f"✅ 검색 완료 - 총 {len(documents)}개 문서")
    print("="*80)
    
    for i, doc in enumerate(documents, 1):
        print(f"\n📑 문서 {i}")
        print("-"*80)
        # 본문 내용 출력
        print("\n📝 본문:")
        content = doc.page_content
        if len(content) > 200:
            content = content[:200] + "..."
        print(content, '\n')
        
        # 메타데이터 출력
        print("📌 메타데이터:")
        metadata = doc.metadata
        for key, value in metadata.items():
            # 테이블 데이터는 특별 처리
            if key == 'table':
                print(f"  ▪ {key}: [표 데이터 포함]")
                continue
            # 긴 텍스트는 축약
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"  ▪ {key}: {value}")
        
        # 테이블 데이터가 있는 경우 표 형식으로 출력
        if 'table' in metadata and metadata['table']:
            print("\n📊 표 데이터:")
            try:
                formatted_table = format_table(metadata['table'])
                if len(formatted_table) > 500:
                    formatted_table = formatted_table[:500] + "..."
                print(formatted_table)
            except:
                print("  [표 데이터 파싱 실패]")
        
        print("-"*80 + "\n")


def preprocess_financial_df(df):
    # 1. 불필요한 열 제거
    columns_to_drop = [
        'ord',
        'fs_nm',
        'bfefrmtrm_nm', 'frmtrm_nm',
        'frmtrm_add_amount',
        'corp_code', 'stock_code', 'reprt_code',
        'fs_div', 'sj_div', 'account_id',
        'account_detail', 'thstrm_add_amount',
    ]
    
    # 연도별 데이터 정리
    df['bsns_year'] = pd.to_numeric(df['bsns_year'], errors='coerce')
    latest_year = df['bsns_year'].max()
    # 전기, 전전기 금액을 숫자로 변환
    df['frmtrm_amount'] = pd.to_numeric(df['frmtrm_amount'], errors='coerce')
    df['bfefrmtrm_amount'] = pd.to_numeric(df['bfefrmtrm_amount'], errors='coerce')
    df['thstrm_amount'] = pd.to_numeric(df['thstrm_amount'], errors='coerce')

    # 연도별로 순회하면서 데이터 정정
    years = sorted(df['bsns_year'].unique(), reverse=True)
    
    for i in range(len(years)):
        current_year = years[i]
        current_year_data = df[df['bsns_year'] == current_year]
        
        # 다음 연도 데이터가 있는 경우
        if i < len(years)-1:
            next_year = years[i+1]
            next_year_data = df[df['bsns_year'] == next_year]
            
            # 현재 연도의 전기 금액과 다음 연도의 당기 금액 비교
            for _, row in current_year_data.iterrows():
                account = row['account_nm']
                prev_amount = row['frmtrm_amount']
                
                next_year_row = next_year_data[next_year_data['account_nm'] == account]
                if not next_year_row.empty:
                    current_amount = next_year_row.iloc[0]['thstrm_amount']
                    if pd.notna(current_amount) and (pd.isna(prev_amount) or prev_amount != current_amount):
                        df.loc[(df['bsns_year'] == current_year) & (df['account_nm'] == account), 'frmtrm_amount'] = current_amount
            
            # 현재 연도의 전전기 금액과 다음 연도의 전기 금액 비교
            for _, row in current_year_data.iterrows():
                account = row['account_nm']
                prev_prev_amount = row['bfefrmtrm_amount']
                
                next_year_row = next_year_data[next_year_data['account_nm'] == account]
                if not next_year_row.empty:
                    prev_amount = next_year_row.iloc[0]['frmtrm_amount']
                    if pd.notna(prev_amount) and (pd.isna(prev_prev_amount) or prev_prev_amount != prev_amount):
                        df.loc[(df['bsns_year'] == current_year) & (df['account_nm'] == account), 'bfefrmtrm_amount'] = prev_amount

    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # 2. 열 이름 한글로 변경
    column_mapping = {
        'rcept_no': '접수번호',
        'corp_code': '기업코드',
        'stock_code': '종목코드',
        'reprt_code': '보고서코드',
        'account_nm': '계정명',
        'fs_div': '재무제표구분',  # CFS/OFS
        'sj_div': '재무제표종류',  # BS/IS
        'sj_nm': '재무제표명',
        'thstrm_dt': '당기일자',
        'thstrm_nm': '당기명',
        'thstrm_amount': '당기금액',
        'frmtrm_dt': '전기일자',
        'frmtrm_amount': '전기금액',
        'bfefrmtrm_dt': '전전기일자',
        'bfefrmtrm_amount': '전전기금액',
        'thstrm_add_amount': '당기누적금액',
        'bsns_year': '연도',
        'currency': '단위',
    }
    
    df = df.rename(columns=column_mapping)
    
    return df

def update_kospi_list():
    """KOSPI 상장 종목 리스트를 최신화하여 CSV 파일로 저장합니다."""
    try:
        # FinanceDataReader를 사용하여 KOSPI 상장 종목 정보 가져오기 (시가총액 내림차순)
        kospi = fdr.StockListing('KOSPI-DESC')
        
        # 컬럼명을 한글로 변경
        kospi_list = kospi.rename(columns={
            'Code': '종목코드',
            'Name': '종목명',
            'Market': '시장구분',
            'Sector': '섹터',
            'Industry': '산업',
            'ListingDate': '상장일',
            'SettleMonth': '결산월',
            'Representative': '대표자명',
            'HomePage': '홈페이지',
            'Region': '지역'
        })
        
        # 종목코드 6자리로 맞추기
        kospi_list['종목코드'] = kospi_list['종목코드'].str.zfill(6)
        
        # data 폴더에 CSV 파일로 저장
        import os
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        file_path = os.path.join(data_dir, 'kospi_list.csv')
        kospi_list.to_csv(file_path, index=False, encoding='utf-8')
        
        print(f"KOSPI 상장 종목 리스트가 성공적으로 저장되었습니다: {file_path}")
        return True
        
    except Exception as e:
        print(f"KOSPI 상장 종목 리스트 업데이트 중 오류 발생: {str(e)}")
        return False

