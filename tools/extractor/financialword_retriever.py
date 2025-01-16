import json
from langchain_community.document_loaders import AsyncChromiumLoader
from bs4 import BeautifulSoup
from typing import Tuple, Optional


class WebScrapeRetriever:
    def __init__(self):
        self.kospi_mapping = self._load_mapping()
    
    def _load_mapping(self) -> dict:
        """Load KOSPI mapping data"""
        with open('tools/retrieve/data/kospi_mapping.json', 'r', encoding='utf-8') as f:
            return json.load(f)

    async def _scrape_data(self, company: str, financial_word: str, year: int) -> Optional[Tuple[str, str]]:
        """Fnguide 데이터를 스크래핑합니다."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        ticker = None
        for comp, code in self.kospi_mapping.items():
            if company == comp:
                ticker = code.split('.')[0]
                break
        
        if not ticker:
            return None

        url = f"https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=A{ticker}&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701"

        try:
            loader = AsyncChromiumLoader([url])
            loader.default_headers = headers
            html = await loader.aload()
            
            if not html or not html[0].page_content:
                return None
                
            soup = BeautifulSoup(html[0].page_content, 'html.parser')
            table = soup.find('table', {'class': 'us_table_ty1 h_fix zigbg_no'})
            if not table:
                return None
            
            headers = table.find('thead')
            if not headers:
                return None
            
            header_cells = headers.find_all('th')
            year_column = None
            for index, header in enumerate(header_cells):
                if header and header.text:
                    header_text = header.text.strip()
                    # 앞의 4글자(연도)만 비교
                    if header_text[:4] == str(year):
                        year_column = index
                        break
            
            if year_column is None:
                return None
            
            tbody = table.find('tbody')
            if not tbody:
                return None
            
            rows = tbody.find_all('tr')
            for row in rows:
                # 모든 텍스트 요소 검사 (th와 td 모두)
                all_cells = row.find_all(['th', 'td'])
                for cell in all_cells:
                    # 셀 내의 모든 텍스트 요소 검사
                    for text_element in cell.find_all(text=True, recursive=True):
                        if text_element.strip() == financial_word:
                            # 해당 행에서 연도에 해당하는 값 찾기
                            td_cells = row.find_all('td')
                            if td_cells and len(td_cells) > year_column - 1:
                                value_cell = td_cells[year_column - 1]
                                if value_cell and value_cell.text:
                                    return value_cell.text.strip(), url
            
            return None
                
        except Exception as e:
            print(f"스크래핑 중 에러 발생: {str(e)}")
            return None

    async def run(self, company: str, financial_word: str, year: int) -> Optional[Tuple[str, str]]:
        """실행 함수"""
        result, url = await self._scrape_data(company, financial_word, year)
        if result is not None:
            return result, url
        return None