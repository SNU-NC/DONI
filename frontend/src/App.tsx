import React, { useState, useRef, useCallback, memo, useEffect } from 'react';
import HTMLFlipBook from 'react-pageflip';
import './App.css';
import { createTypeStream } from 'hangul-typing-animation';

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

declare global {
    interface Window {
        TypeHangul: any;
        Hangul: any;
        Typed: any;
    }
}

const ThinkingMessage = memo(() => {
    const [thinkingIndex, setThinkingIndex] = useState(0);

    const thinkingMessages = [
        "💡 주식투자는 장기적 관점이 중요해요",
        "💡 투자 결정 전 항상 재무제표를 확인하세요",
        "💡 시장 전체의 흐름을 파악하는 것이 중요해요",
        "💡 질문 시, 원하는 기업과 주제, 연도를 명확히 하면 더 정확한 답변을 얻을 수 있어요",
        "💡 현재 코스피 기업에 대해서 정보를 제공하고 있습니다. 업데이트를 기다려주세요 😉",
        "데이터 수집 중... 📥",
        "분석 중... 🔍",
        "답변 생성 중... ✍️",
        "마무리 중... 🎯",
 
    ];

    useEffect(() => {
        const interval = setInterval(() => {
            setThinkingIndex((prev) => (prev + 1) % thinkingMessages.length);
        }, 2000); // 2초마다 메시지 변경
        
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="message-container">
            <div className="avatar">
                <img src="/component/imgs/robot_avatar.png" alt="AI 챗봇" />
            </div>
            <div className="message bot-message thinking">
                {thinkingMessages[thinkingIndex]}
            </div>
        </div>
    );
});

const MessageItem = memo(({ msg, idx, isNew }: { msg: Message; idx: number; isNew?: boolean }) => {
    const messageRef = useRef<HTMLDivElement>(null);
    const typeStreamRef = useRef<any>(null);

    useEffect(() => {
        if (!msg.isUser && isNew && messageRef.current) {
            const typeStream = createTypeStream({
                perChar: 20,    // 일반 문자 타이핑 속도
                perHangul: 0,  // 한글 타이핑 속도
                perSpace: 0,    // 공백 타이핑 속도
                perLine: 0,     // 줄바꿈 타이핑 속도
                perDot: 0     // 마침표 타이핑 속도
            });

            const timer = setTimeout(() => {
                if (messageRef.current) {
                    typeStreamRef.current = typeStream(msg.content, (typing) => {
                        if (messageRef.current) {
                            messageRef.current.textContent = typing;
                        }
                    });
                }
            }, 500);

            return () => {
                clearTimeout(timer);
            };
        } else if (messageRef.current) {
            messageRef.current.textContent = msg.content;
        }
    }, [msg.isUser, isNew, msg.content]);

    return (
        <div className={`message-container ${msg.isUser ? 'user' : ''} ${isNew ? 'new-message' : ''}`}>
            <div className={`avatar ${msg.isUser ? 'user' : ''}`}>
                {msg.isUser ? '👤' : <img src="/component/imgs/robot_avatar.png" alt="AI 챗봇" />}
            </div>
            <div 
                ref={!msg.isUser ? messageRef : null}
                className={`message ${msg.isUser ? 'user-message' : 'bot-message'}`}
            >
                {msg.isUser ? msg.content : ''}
            </div>
            <div className="timestamp">{msg.timestamp}</div>
        </div>
    );
});

const ChatMessages = memo(({ messages, isThinking }: { messages: Message[]; isThinking: boolean }) => {
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const [isInitialLoad, setIsInitialLoad] = useState(true);
    const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
    const lastMessageRef = useRef<HTMLDivElement>(null);
    
    // 스크롤 위치 감지
    // useCallback 사용 이유: 함수 재생성 방지 인자로 전달한 함수 자체를 메모라이제이션 
    const handleScroll = useCallback(() => {
        if (!chatContainerRef.current) return;
        
        const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
        const isNearBottom = scrollHeight - (scrollTop + clientHeight) < 100;
        setShouldAutoScroll(isNearBottom);
    }, []);

    useEffect(() => {
        const container = chatContainerRef.current;
        if (container) {
            container.addEventListener('scroll', handleScroll);
            return () => container.removeEventListener('scroll', handleScroll);
        }
    }, [handleScroll]);

    // 새 메시지가 추가될 때 스크롤
    useEffect(() => {
        if (isInitialLoad) {
            if (chatContainerRef.current) {
                chatContainerRef.current.scrollTop = 0;
            }
            setIsInitialLoad(false);
        } else if (shouldAutoScroll && messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    }, [messages, isInitialLoad, shouldAutoScroll]);

    return (
        <div className="chat-messages" ref={chatContainerRef}>
            <div className="messages-wrapper">
                {messages.map((msg, idx) => (
                    <div 
                        key={msg.timestamp + '-' + idx}
                        ref={idx === messages.length - 1 ? lastMessageRef : null}
                    >
                        <MessageItem 
                            msg={msg} 
                            idx={idx}
                            isNew={idx === messages.length - 1 && !isInitialLoad}
                        />
                    </div>
                ))}
                {isThinking && <ThinkingMessage />}
                <div ref={messagesEndRef} style={{ height: '20px' }} />
            </div>
        </div>
    );
});

const ChatPage = memo(React.forwardRef<HTMLDivElement, { messages: Message[]; isThinking: boolean }>((props, ref) => {
    return (
        <div className="page chat-page" ref={ref}>
            <div className="page-header">
                <h2>💬 대화</h2>
            </div>
            <ChatMessages messages={props.messages} isThinking={props.isThinking} />
        </div>
    );
}));

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

const InputComponent = memo(({ onSend, isThinking }: { onSend: (value: string) => void, isThinking: boolean }) => {
    const [inputValue, setInputValue] = useState('');

    const handleSend = useCallback(() => {
        if (!inputValue.trim()) return;
        onSend(inputValue);
        setInputValue('');
    }, [inputValue, onSend]);

    return (
        <div className="input-container">
            <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder="질문을 입력하세요..."
                disabled={isThinking}
            />
            <button onClick={handleSend} disabled={isThinking}>
                {isThinking ? '전송' : '전송'}
            </button>
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
    const [isThinking, setIsThinking] = useState(false);
    const bookRef = useRef<any>();

    const handleSendMessage = useCallback(async (inputValue: string) => {
        const timestamp = new Date().toLocaleTimeString('ko-KR');
        const newMessage: Message = {
            content: inputValue,
            isUser: true,
            timestamp
        };

        setMessages(prev => [...prev, newMessage]);
        setIsThinking(true);
        
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
    }, []);

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
            
            <InputComponent onSend={handleSendMessage} isThinking={isThinking} />
        </div>
    );
}

export default App; 