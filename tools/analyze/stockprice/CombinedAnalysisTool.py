from tools.analyze.stockprice.CompanyAnalyzerTool import CompanyAnalyzerTool
from tools.analyze.stockprice.DCFTool import DCFTool
from tools.analyze.stockprice.NowstockPriceTool import NowStockPriceTool
from typing import Type
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from langchain_core.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from config.prompts import _CombinedAnalysisTool_DESCRIPTION


# 입력 스키마 정의
class CombinedToolInputSchema(BaseModel):
    query: str = Field(..., description="검색하고자 하는 문장")
    company: str = Field(..., description="회사명")
    year: int = Field(..., description="연도")
    class Config:
        extra = "allow" # 추가 필드 허용
# """
# CompanyAnalyzer(company_name:str) -> str:
#  - company_name은 예측하고자 하는 기업 이름입니다.
#  - CompanyAnalyzer 클래스는 기업의 가치평가에 대한 질문이 들어왔을 때, 검색 기업의 업종 정보를 추출하고, EPS, BPS, PER, PBR, ROE등 재무지표 추출, 업종 내 회귀분석을 통한 PER/PBR 예측, 그리고 최종적으로 가치 평가를  수행합니다.
#  - 결과를 문자열 형태로 반환합니다.
#  - 기업의 가치평가에 대한 질문이 들어왔을 때 이 Tool을 사용합니다. 
# """

# _DCF_DESCRIPTION = """
# DCF(company_name: str) -> str:
# - company_name**: 가치 평가를 수행할 기업의 이름입니다.
# - 기업의 FCFE, WACC, 순이익 성장률 등 재무 데이터를 활용하여 10년 미래 현금흐름과 터미널 가치를 계산하고, 이를 통해 기업의 총 가치와 주당 가치를 산출합니다. 
# - 결과를 문자열로 반환합니다.
# - 기업의 내재 가치 평가나 DCF 모델을 사용한 주당 가치 분석이 필요할 때 이 Tool을 사용합니다.
# """

# _STOCKPRICE_DESCRIPTION="""
# NowStockPriceTool(company_name:str)->int:
# -company_name은 예측하고자 하는 기업 이름입니다.
# -기업의 현재 주식가격을 예측해주는 툴이고 결과를 정수 형태로 반환합니다
# -PER,PBR,DCF기반 가치 예측을 한 후 현재 주식가격과 비교하여 저평가 되었는지 고평가 되었는지 분석할 때 쓰입니다.
# -PER,PBR,DCF기반 가치 예측을 한 후 기업의 주가의 방향을 예측할 때 쓰입니다.
# """


# 통합 툴 정의
class CombinedAnalysisTool(BaseTool):
    """CompanyAnalyzerTool, DCFTool, NowStockPriceTool을 통합한 툴"""
    
    name: str = "CombinedAnalysisTool"
    # description: str = ("CombinedAnalysisTool(query:str) -> str:"
    #     "이 툴은 특정 기업의 재무 상태와 가치를 종합적으로 분석하기 위한 통합 도구입니다.\n"
    #     "1. **CompanyAnalyzerTool**: 기업의 업종 정보를 추출하고, EPS, BPS, PER, PBR, ROE와 같은 재무 지표를 분석하며, 업종 내 회귀분석을 통해 PER 및 PBR을 예측하고 최종적으로 기업의 가치를 평가합니다.\n"
    #     "2. **DCFTool**: 기업의 FCFE, WACC, 순이익 성장률을 바탕으로 10년 미래 현금흐름과 터미널 가치를 계산하고, 이를 통해 기업의 총 가치와 주당 가치를 산출합니다.\n"
    #     "3. **NowStockPriceTool**: 현재 주식 가격을 추출하여, PER, PBR, DCF 기반 가치 예측 결과와 비교해 기업이 저평가 또는 고평가 상태인지 분석합니다.\n"
    #     "이 툴은 기업의 종합적인 재무 분석을 수행하고, 가치를 평가하며, 현재 주가와의 비교를 통해 투자 의사 결정을 지원합니다."
    #     "PER 기반 예측에서는, 동일 업계 기업들의 EPS(주당순이익) 평균 성장률과 베타 값을 이용하여 회귀 분석을 통해 PER(주가수익비율)을 추정하였으며, 추정된 PER에 대상 기업의 EPS를 곱하여 기업 가치를 계산하였음을 설명하세요."
    #     "PBR 기반 예측에서는, 동일 업계 기업들의 ROE(자기자본이익률)와 베타 값을 이용하여 회귀 분석을 통해 PBR(주가순자산비율)을 추정하였으며, 추정된 PBR에 대상 기업의 BPS(주당순자산)를 곱하여 기업 가치를 계산하였음을 설명하세요."
    #     "또한, PER 및 PBR 기반 예측을 수행하는 도구가 정확한 수치 예측을 제공하지 못하는 경우, 도구에서 반환된 응답을 분석하여 왜 예측이 불가능한지에 대한 이유를 포함하세요."
    #     "이와 더불어, PER 및 PBR 예측 방식이 기업의 PER과 PBR이 최근 성장 실적과 시장 대비 위험에 의해 결정되며 평균으로 회귀하는 경향이 있다는 가정에 기반하고 있음을 설명하세요."
    #     "DCF(Discounted Cash Flow) 기반 분석에서는 다음 단계를 자세히 설명하세요:"
    #     "1.	과거 재무 데이터를 기반으로 미래 현금 흐름을 예측하여 **자기자본 잉여현금흐름(FCFE, Free Cash Flow to Equity)**를 계산하였습니다."
    #     "2.	기업의 베타 값, 시장 위험 프리미엄, 무위험 이자율을 고려하여 **자기자본 비용(Cost of Equity)**을 결정하였습니다."
    #     "3.	순이익 성장률과 일치하는 성장을 가정하여 향후 10년간 예측된 FCFE의 현재 가치를 계산하였습니다."
    #     "4.	“**10년 이후 안정 상태의 성장을 추정하여 영구 현금 흐름(Present Value of Perpetuity)**의 현재 가치를 계산하였습니다."
    #     "5.	“예측된 FCFE의 10년치 현재 가치와 영구 현금 흐름의 현재 가치를 합산하여 총 가치를 계산한 후, 총 발행 주식 수로 나누어 **주당 가치(Value per Share)**를 산출하였습니다."
    # )
    description: str = _CombinedAnalysisTool_DESCRIPTION
        # "1. **CompanyAnalyzerTool**: 기업의 업종 정보를 추출하고, EPS, BPS, PER, PBR, ROE와 같은 재무 지표를 분석하며, 업종 내 회귀분석을 통해 PER 및 PBR을 예측하고 최종적으로 기업의 가치를 평가합니다.\n"
        # "2. **DCFTool**: 기업의 FCFE, WACC, 순이익 성장률을 바탕으로 10년 미래 현금흐름과 터미널 가치를 계산하고, 이를 통해 기업의 총 가치와 주당 가치를 산출합니다.\n"
        # "3. **NowStockPriceTool**: 현재 주식 가격을 추출하여, PER, PBR, DCF 기반 가치 예측 결과와 비교해 기업이 저평가 또는 고평가 상태인지 분석합니다.\n"
  
    args_schema: Type[BaseModel] = CombinedToolInputSchema
    return_direct: bool = True

    # 클래스 속성 정의
    company_analyzer: CompanyAnalyzerTool = Field(default_factory=CompanyAnalyzerTool)
    dcf_tool: DCFTool = Field(default_factory=DCFTool)
    now_stock_price_tool: NowStockPriceTool = Field(default_factory=NowStockPriceTool)

    def _run(self, query: str, company:str, year:int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:              

            results = {}

            # CompanyAnalyzerTool 실행
            #print("Running CompanyAnalyzerTool...")
            results["valuation"] = self.company_analyzer.run({"company_name": company})

            # NowStockPriceTool 실행
            #print("Running NowStockPriceTool...")
            results["stock_price"] = self.now_stock_price_tool.run({"company_name": company})

            # DCFTool 실행
            #print("Running DCFTool...")
            results["dcf"] = self.dcf_tool.run({"company_name": company})

            return results
        except Exception as e:
            e
        raise RuntimeError(f"CombinedAnalysisTool execution failed: {e}")