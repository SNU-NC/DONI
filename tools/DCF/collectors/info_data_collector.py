import yfinance as yf

class InfoDataCollector:
    """info 데이터를 수집하는 클래스"""
    
    def __init__(self, ticker_symbol: str):
        self.ticker_symbol = ticker_symbol
        self.stock = yf.Ticker(ticker_symbol)
    
    def get_info(self) -> dict:
        """info 데이터를 가져오는 메서드
        
        returns:
            dict: info 딕셔너리
        """

        info = self.stock.info
        info_metrics = self._get_info_metrics(info)
            
        return info_metrics
    


    def _get_info_metrics(self, info: dict) -> dict:
        """재무상태표 관련 지표 추출"""
        metrics = {}
        
        metrics['payout_ratio'] = info.get('payoutRatio', 0)
        if metrics['payout_ratio'] == 0:
            print("payoutRatio 데이터 누락: 기본값으로 대체.")
        
        metrics['regularMarketPreviousClose'] = info.get('regularMarketPreviousClose', None)
        if metrics['regularMarketPreviousClose'] is None:
            print("regularMarketPreviousClose 데이터를 찾을 수 없습니다.")
        
        metrics['shares_outstanding'] = info.get('sharesOutstanding', None)
        if metrics['shares_outstanding'] is None:
            print("sharesOutstanding 데이터를 찾을 수 없습니다.")
        
        return metrics
    
    # def _get_info_metrics(self, info: dict) -> dict:
    #     """재무상태표 관련 지표 추출"""
    #     metrics = {}
        
    #     try:
    #         metrics['payout_ratio'] = info['payoutRatio']
    #     except KeyError:
    #         metrics['payout_ratio'] = 0
    #         print("payoutRatio 데이터를 찾을 수 없습니다. 배당이 없던 것으로 간주하여, payout_ratio = 0으로 대체합니다.")

    #     try:
    #         metrics['regularMarketPreviousClose'] = info['regularMarketPreviousClose']
    #     except KeyError:
    #         metrics['regularMarketPreviousClose'] = 0
    #         print("regularMarketPreviousClose 데이터를 찾을 수 없습니다.")

    #     try:
    #         metrics['shares_outstanding'] = info['sharesOutstanding']
    #     except KeyError:
    #         metrics['shares_outstanding'] = 1
    #         print("sharesOutstanding 데이터를 찾을 수 없습니다.")
        
    #     return metrics