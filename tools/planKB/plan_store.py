from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
import json
import os
from tools.planKB.plan_model import PlanExample

class PlanStore:
    def __init__(self, embedding_model: Embeddings):
        self.embedding_model = embedding_model
        self.vector_store = None
        self._initialize_store("tools/planKB/data/plan_examples.json")
        
    def _initialize_store(self, plans_file: str):
        """계획 예시들을 로드하고 벡터스토어 초기화"""
        if not os.path.exists(plans_file):
            raise FileNotFoundError(f"Plans file not found: {plans_file}")
            
        try:
            with open(plans_file, 'r', encoding='utf-8') as f:
                try:
                    plans = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in plans file: {e}")
                    
        except IOError as e:
            raise IOError(f"Error reading plans file: {e}")
            
        try:
            texts = [plan['query'] for plan in plans]
            metadatas = []
            for plan in plans:
                metadata = {
                    'plan': json.dumps({
                        'steps': plan['plan']['steps']
                    }, ensure_ascii=False)
                }
                
                # extra_info가 있는 경우에만 메타데이터에 추가
                if 'extra_info' in plan and plan['extra_info'] is not None:
                    metadata['extra_info'] = plan['extra_info']
                    
                metadatas.append(metadata)

            self.vector_store = Chroma(
                collection_name="plan_examples",
                embedding_function=self.embedding_model
            )
            # 쿼리와 관련 계획을 메타데이터로 저장
            self.vector_store.add_texts(texts, metadatas=metadatas)
            
        except Exception as e:
            raise RuntimeError(f"Error initializing vector store: {e}")
        
    def get_similar_examples(self, query: str, k: int = 1, score_threshold: float = 1 ) -> List[PlanExample]:
        """쿼리와 유사한 계획 예시들 검색
        Args:
            query: 검색할 쿼리
            k: 반환할 결과 수
            score_threshold: 낮은 점수가 가까운 거임 
        Returns:
            List[PlanExample]: 유사도 기준을 만족하는 계획 예시들
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")   

        def create_plan_example(doc, score: float ) -> Optional[PlanExample]:
            """단일 문서를 PlanExample로 변환"""
            try:
                if score > score_threshold :
                    print(f"Skipping example with low similarity score: {score:.3f}")
                    print(doc.page_content)
                    return None 
                else :
                    print("별로 비슷하지 않음 ")
                    print(score)
                    print(doc.page_content)
                    print(doc.metadata)


                example = PlanExample(
                    query=doc.page_content,
                    plan=json.loads(doc.metadata['plan']),
                    extra_info = doc.metadata.get('extra_info', None)
                )
                print(f"Found similar example with score: {score:.3f}")
                return example
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing document: {e}")
                return None

        # list comprehension과 filter를 사용하여 코드 단순화
        examples = [
            example for doc, score in self.vector_store.similarity_search_with_score(query, k=k)
            if (example := create_plan_example(doc, score)) is not None
        ]
        
        return examples 