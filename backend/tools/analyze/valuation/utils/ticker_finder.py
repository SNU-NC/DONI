from deep_translator import GoogleTranslator
from yahooquery import search
from .korean_checker import contains_korean

class TickerFinder:
    @staticmethod
    def get_ticker(company_name: str) -> str:
        """회사명으로 티커를 검색합니다."""
        try:
            is_korean = contains_korean(company_name)
            search_term = (GoogleTranslator(source='auto', target='en').translate(company_name) 
                         if is_korean else company_name)
            
            results = search(search_term)
            
            exchange_priority = {
                'KSC': 1,  # 한국
                'NYQ': 2,  # NYSE
                'NMS': 3,  # NASDAQ
                'JPX': 4,  # 일본
                'HKG': 5   # 홍콩
            }
            
            valid_quotes = [quote for quote in results['quotes'] 
                          if quote['exchange'] in exchange_priority]
            
            if not valid_quotes:
                return None
                
            return min(valid_quotes, 
                      key=lambda x: exchange_priority[x['exchange']])['symbol']
            
        except Exception as e:
            print(f"Error finding ticker for {company_name}: {e}")
            return None 