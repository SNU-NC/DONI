import pandas as pd

class predictNextQuarter:
    def __init__(self, income_stmt_cum:pd.DataFrame, state, segments):
        self.income_stmt_cum = income_stmt_cum
        self.state = state
        self.segments = segments
    
    def fill_next_quarter_df(self):
        # 다음 분기(예측) 열 추가 (모든 행 0으로 초기화)
        temp_df = self.income_stmt_cum.copy()
        temp_df['next_quarter'] = 0

        for segment in self.segments:
            yoy_prediction = self.state.segment[segment]['yoy_prediction']
            predicted_growth_rate = 1 + (yoy_prediction / 100)
            
            # 해당 segment 행의 next_quarter 값을 업데이트(next_quarter의 1년 전 분기 값 * 성장률)
            temp_df.loc[temp_df['계정'] == segment, 'next_quarter'] = temp_df.loc[temp_df['계정'] == segment].iloc[:,2] * predicted_growth_rate
        
        # next_quarter의 segments 행들의 합계 계산
        revenue_sum = temp_df.loc[temp_df['계정'].isin(self.segments), 'next_quarter'].sum()

        # segments의 합계를 영업수익 값으로 업데이트
        temp_df.loc[temp_df['계정'] == '영업수익', 'next_quarter'] = revenue_sum

        # 직전분기의 영업수익 대비 영업이익 비율 계산
        revenue_current_quarter = temp_df.loc[temp_df['계정'] == '영업수익'].iloc[:,5].values[0]
        operating_profit_current_quarter = temp_df.loc[temp_df['계정'] == '영업이익'].iloc[:,5].values[0]
        operating_profit_margin = operating_profit_current_quarter / revenue_current_quarter

        # 동기간 전분기의 영업이익율을 다음 분기의 영업이익 예측에 사용(영업이익률을 그대로 사용)
        revenue_next_quarter = temp_df.loc[temp_df['계정'] == '영업수익', 'next_quarter'].values[0]
        operating_profit_next_quarter = revenue_next_quarter * operating_profit_margin

        # 영업이익 행의 next_quarter 값을 업데이트
        temp_df.loc[temp_df['계정'] == '영업이익', 'next_quarter'] = operating_profit_next_quarter

        # 직전분기의 순이익률 계산
        net_income_current_quarter = temp_df.loc[temp_df['계정'] == '순이익'].iloc[:,5].values[0]
        net_margin_current_quarter = net_income_current_quarter / revenue_current_quarter

        # 다음분기의 순이익 계산(직전분기의 순이익률 * 다음분기의 추정 영업수익)
        net_income_next_quarter = revenue_next_quarter * net_margin_current_quarter

        # 순이익 행 업데이트
        temp_df.loc[temp_df['계정'] == '순이익', 'next_quarter'] = net_income_next_quarter
                
        return temp_df
                