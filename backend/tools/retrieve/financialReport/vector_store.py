"""
벡터 스토어 초기화 및 관리 모듈
"""

import os
import json
from typing import List, Optional
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import ClovaXEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import pickle
from pathlib import Path

class VectorStoreManager:
    """벡터 스토어 관리 클래스"""
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 초기화 플래그 추가
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, persist_dir: str = "./chroma_db", cache_dir: str = "./embedding_cache"):
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
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings = ClovaXEmbeddings(
            service_app=True,
            model_name="v2",
            timeout=60
        )
        
        # 초기화 완료 표시
        self._initialized = True
        
    def load_or_create(self, create_flag: bool = False) -> Chroma:
        if not create_flag:
            print("기존 벡터스토어를 로드합니다...")
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            
        print("벡터스토어를 구축합니다...")
        return self._create_new_vectorstore()
    
    def _create_new_vectorstore(self) -> Chroma:
        """새로운 벡터스토어 생성"""
        docs = self._load_documents()
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        
        self._add_documents_with_retry(vectorstore, docs)
                
        return vectorstore
    
    def _load_documents(self) -> List[Document]:
        """JSON 파일에서 새로운 문서만 로드"""
        docs = []
        json_dir = "./data/docs"
        
        # 기존 ChromaDB에서 메타데이터 가져오기
        existing_metadata = {}
        if os.path.exists(self.persist_dir):
            vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            collection_data = vectorstore._collection.get()
            for doc_id, metadata in zip(collection_data['ids'], collection_data['metadatas']):
                key = f"{metadata.get('corp_code')}_{metadata.get('report_id')}"
                existing_metadata[key] = doc_id
        
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        print(f"총 {len(json_files)}개의 JSON 파일을 확인합니다...")
        
        new_doc_count = 0
        for filename in tqdm(json_files, desc="JSON 파일 검사 중"):
            with open(os.path.join(json_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    metadata = self._filter_metadata(item.get('metadata', {}))
                    doc_key = f"{metadata.get('corp_code')}_{metadata.get('report_id')}"
                    
                    # 새로운 문서인 경우에만 추가
                    if doc_key not in existing_metadata:
                        doc = Document(
                            page_content=item.get('page_content', ''),
                            metadata=metadata
                        )
                        docs.append(doc)
                        new_doc_count += 1
        
        print(f"새로 추가될 문서 수: {new_doc_count}")
        return docs
    
    @staticmethod
    def _filter_metadata(metadata: dict) -> dict:
        """메타데이터 필터링"""
        filtered = {}
        for key, value in metadata.items():
            if isinstance(value, (str, bool, int, float)):
                filtered[key] = value
            elif key == 'table' and isinstance(value, list):
                filtered[key] = json.dumps(value, ensure_ascii=False)
        return filtered
    
    def _generate_batch_id(self, batch: List[Document]) -> str:
        """배치의 고유 ID 생성"""
        # 문서 내용을 기반으로 해시 생성
        content = "".join(doc.page_content for doc in batch)
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

    def _load_cache(self, batch_id: str) -> Optional[List[List[float]]]:
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

    def _add_embeddings_to_store(
        self,
        vectorstore: Chroma,
        docs: List[Document],
        ids: List[str],
        embeddings: List[List[float]],
        batch_id: str
    ) -> None:
        """생성된 임베딩을 벡터 스토어에 추가하고 캐시에 저장"""
        vectorstore._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs]
        )
        self._save_cache(batch_id, embeddings)
        
    def _add_documents_with_retry(self, vectorstore: Chroma, docs: List[Document]) -> None:
        """문서 추가 재시도 로직 (캐싱 포함)"""
        batch_size = 100
        for i in tqdm(range(0, len(docs), batch_size), desc="문서 처리 중"):
            batch = docs[i:i + batch_size]
            batch_id = self._generate_batch_id(batch)
            
            # 각 문서의 ID 생성
            ids = [f"doc_{i+j}" for j in range(len(batch))]
            
            # 기존 문서 확인
            existing_docs = vectorstore.get(ids=ids)
            existing_ids = set(existing_docs["ids"])
            
            # 새로운 문서만 필터링
            new_docs = []
            new_ids = []
            for doc, doc_id in zip(batch, ids):
                if doc_id not in existing_ids:
                    new_docs.append(doc)
                    new_ids.append(doc_id)
            
            if not new_docs:  # 새로운 문서가 없으면 건너뛰기
                continue
            
            # 캐시 확인
            cached_embeddings = self._load_cache(batch_id)
            
            if cached_embeddings:
                # Chroma의 내부 메서드를 사용하여 캐시된 임베딩 직접 추가
                vectorstore._collection.add(
                    ids=new_ids,
                    embeddings=cached_embeddings[:len(new_docs)],  # 새 문서 수만큼만 사용
                    documents=[doc.page_content for doc in new_docs],
                    metadatas=[doc.metadata for doc in new_docs]
                )
            else:
                embeddings = self._create_embeddings_with_retry([doc.page_content for doc in new_docs])
                self._add_embeddings_to_store(
                    vectorstore,
                    new_docs,
                    new_ids,
                    embeddings,
                    batch_id
                )