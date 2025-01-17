import React, { useState, useRef, useCallback } from 'react';
import HTMLFlipBook from 'react-pageflip';
import './App.css';

interface Message {
    content: string;
    isUser: boolean;
    timestamp: string;
}

interface Reference {
    tool?: string;
    referenced_content?: string;
    filename?: string;
    page_number?: number;
    link?: string;
    title?: string;
    broker?: string;
    target_price?: string;
    investment_opinion?: string;
    analysis_result?: string;
    type?: string;
    content?: string;
}

const ThinkingMessage = () => (
    <div className="message-container">
        <div className="avatar">
            <img src="/component/imgs/robot_avatar.png" alt="AI 챗봇" />
        </div>
        <div className="message bot-message thinking">
            생각 중
            <span className="dots"></span>
        </div>
    </div>
);

const ChatPage = React.forwardRef<HTMLDivElement, { messages: Message[]; isThinking: boolean }>((props, ref) => {
    return (
        <div className="page chat-page" ref={ref}>
            <div className="page-header">
                <h2>💬 대화</h2>
            </div>
            <div className="chat-messages">
                {props.messages.map((msg, idx) => (
                    <div key={`${msg.timestamp}-${idx}`} className={`message-container ${msg.isUser ? 'user' : ''}`}>
                        <div className={`avatar ${msg.isUser ? 'user' : ''}`}>
                            {msg.isUser ? '👤' : <img src="/component/imgs/robot_avatar.png" alt="AI 챗봇" />}
                        </div>
                        <div className={`message ${msg.isUser ? 'user-message' : 'bot-message'}`}>
                            {msg.content}
                        </div>
                        <div className="timestamp">{msg.timestamp}</div>
                    </div>
                ))}
                {props.isThinking && <ThinkingMessage />}
            </div>
        </div>
    );
});

const ReferencePage = React.forwardRef<HTMLDivElement, { references: Reference[] }>((props, ref) => {
    const typeIcons = {
        'analyst_report': '📊',
        'financial_report': '📑',
        'market_analysis': '📈',
        'reference': '📚',
        'web_search': '🌐',
        'other': '📄'
    };

    const getDisplayContent = (ref: Reference) => {
        let content = '';
        let details: string[] = [];

        if (ref.referenced_content) {
            content = ref.referenced_content;
        } else if (ref.analysis_result) {
            content = ref.analysis_result;
        } else if (ref.content) {
            content = ref.content;
        }
        
        if (ref.broker) {
            details.push(`${ref.broker}`);
        }
        if (ref.target_price) {
            details.push(`목표가: ${ref.target_price}`);
        }
        if (ref.investment_opinion) {
            details.push(`투자의견: ${ref.investment_opinion}`);
        }
        
        return {
            mainContent: content,
            details: details.join(' | ')
        };
    };

    const getDisplayTitle = (ref: Reference) => {
        if (ref.title) return ref.title;
        if (ref.filename) return ref.filename;
        return ref.tool || '참고자료';
    };

    const getDisplayType = (ref: Reference) => {
        if (ref.tool?.includes('analyst')) return 'analyst_report';
        if (ref.tool?.includes('financial')) return 'financial_report';
        if (ref.tool?.includes('market')) return 'market_analysis';
        if (ref.tool?.includes('web')) return 'web_search';
        return 'other';
    };

    return (
        <div className="page reference-page" ref={ref}>
            <div className="page-header">
                <h2>📚 참고 자료</h2>
            </div>
            <div className="references-container">
                {props.references.map((ref, idx) => {
                    const displayType = getDisplayType(ref);
                    const displayTitle = getDisplayTitle(ref);
                    const {mainContent, details} = getDisplayContent(ref);
                    
                    return (
                        <div key={idx} className="reference-item">
                            <div className="reference-title">
                                {typeIcons[displayType as keyof typeof typeIcons]} {displayTitle}
                                {ref.page_number && ` (p.${ref.page_number})`}
                            </div>
                            {details && <div className="reference-details">{details}</div>}
                            <div className="reference-content">{mainContent}</div>
                            {ref.link && (
                                <a href={ref.link} target="_blank" rel="noopener noreferrer" className="reference-link">
                                    원문 보기
                                </a>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
});

function App() {
    const [messages, setMessages] = useState<Message[]>([{
        content: "안녕하세요! 금융 정보 검색 AI 챗봇입니다. 궁금하신 내용을 자유롭게 질문해 주세요.",
        isUser: false,
        timestamp: new Date().toLocaleTimeString()
    }]);
    const [references, setReferences] = useState<Reference[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isThinking, setIsThinking] = useState(false);
    const bookRef = useRef<any>();

    const handleSendMessage = async () => {
        if (!inputValue.trim()) return;

        const newMessage: Message = {
            content: inputValue,
            isUser: true,
            timestamp: new Date().toLocaleTimeString('ko-KR')
        };

        setMessages(prev => [...prev, newMessage]);
        setInputValue('');
        setIsThinking(true);
        setReferences([]);

        try {
            const response = await fetch('http://localhost:8000/api/search', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ query: inputValue })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setIsThinking(false);

            if (data.error) {
                setMessages(prev => [...prev, {
                    content: data.message,
                    isUser: false,
                    timestamp: new Date().toLocaleTimeString('ko-KR')
                }]);
            } else {
                setMessages(prev => [...prev, {
                    content: data.answer,
                    isUser: false,
                    timestamp: new Date().toLocaleTimeString('ko-KR')
                }]);
                
                if (data.docs && data.docs.length > 0) {
                    setReferences(data.docs);
                    setTimeout(() => {
                        bookRef.current?.pageFlip().flipNext();
                    }, 1000);
                }
            }
        } catch (error) {
            console.error('API 요청 오류:', error);
            setIsThinking(false);
            setMessages(prev => [...prev, {
                content: "죄송합니다. 서버 연결에 실패했습니다. 잠시 후 다시 시도해 주세요.",
                isUser: false,
                timestamp: new Date().toLocaleTimeString('ko-KR')
            }]);
        }
    };

    return (
        <div className="app-container">
            <HTMLFlipBook
                width={600}
                height={800}
                size="stretch"
                minWidth={300}
                maxWidth={1000}
                minHeight={400}
                maxHeight={1000}
                maxShadowOpacity={0.5}
                showCover={false}
                mobileScrollSupport={true}
                className="book"
                ref={bookRef}
                style={{}}
                startPage={0}
                drawShadow={true}
                flippingTime={1000}
                usePortrait={true}
                startZIndex={0}
                autoSize={true}
                clickEventForward={true}
                useMouseEvents={true}
                swipeDistance={30}
                showPageCorners={true}
                disableFlipByClick={false}
            >
                <ChatPage messages={messages} isThinking={isThinking} />
                <ReferencePage references={references} />
            </HTMLFlipBook>
            
            <div className="input-container">
                <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder="질문을 입력하세요..."
                    disabled={isThinking}
                />
                <button onClick={handleSendMessage} disabled={isThinking}>
                    {isThinking ? '생각 중...' : '전송'}
                </button>
            </div>
        </div>
    );
}

export default App; 