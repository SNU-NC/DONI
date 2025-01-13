"""
리포트 벡터 스토어 초기화 및 관리 모듈
"""

import os
import logging
from typing import Tuple
from langchain_chroma import Chroma
from langchain_community.embeddings import ClovaXEmbeddings

class VectorStoreManager:
    """리포트 벡터 스토어 관리 클래스"""
    
    def __init__(self, persist_dir: str = "./chroma_db_02"):
        """
        Args:
            persist_dir: 벡터 스토어 저장 경로
        """
        self.persist_dir = persist_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # ClovaX 임베딩 초기화
        self.embeddings = ClovaXEmbeddings(
            service_app=True,
            model_name="v2",
            timeout=60
        )
        
    def load_or_create(self, create_flag: bool = False) -> Tuple[Chroma, bool]:
        """기존 벡터스토어 로드 또는 새로 생성
        
        Returns:
            Tuple[Chroma, bool]: (vectorstore, is_new_store)
                - vectorstore: Chroma 벡터스토어 인스턴스
                - is_new_store: 새로 생성된 스토어인지 여부
        """
        try:
            db_exists = os.path.exists(os.path.join(self.persist_dir, "chroma.sqlite3"))
            
            if not db_exists and not create_flag:
                raise ValueError(f"벡터스토어가 {self.persist_dir}에 존재하지 않습니다. create_flag=True로 설정하여 새로 생성하세요.")
            
            if db_exists:
                print("기존 벡터스토어2️⃣를 로드합니다...")
                vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                    collection_name="analyst_report"
                )
                return vectorstore, False
            
            if create_flag:
                print("새로운 벡터스토어2️⃣를 생성합니다...")
                os.makedirs(self.persist_dir, exist_ok=True)
                vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                    collection_name="analyst_report"
                )
                return vectorstore, True
            
        except Exception as e:
            self.logger.error(f"벡터스토어2️⃣ 초기화 오류: {str(e)}")
            raise