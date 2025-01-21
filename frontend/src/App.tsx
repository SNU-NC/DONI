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

interface PlanStep {
    tool: string;
    description: string;
    status: 'pending' | 'running' | 'completed';
}

interface TaskProgress {
    type: string;
    status: string;
    task_id?: string;
    task_name?: string;
    result?: any;
    debug_info?: any;
    timestamp: string;
    task_count?: number;
    results?: any[];
    tool_type?: string;
}

interface TaskEvent {
    type: string;
    data: TaskProgress;
}

const typeIcons = {
    'analyst_report': '📊',
    'financial_report': '📑',
    'market_analysis': '📈',
    'reference': '📚',
    'web_search': '🌐',
    'other': '📄'
};

const getDisplayType = (ref: Reference) => {
    if (ref.tool?.includes('analyst')) return 'analyst_report';
    if (ref.tool?.includes('financial')) return 'financial_report';
    if (ref.tool?.includes('market')) return 'market_analysis';
    if (ref.tool?.includes('web')) return 'web_search';
    return 'other';
};

const getDisplayTitle = (ref: Reference) => {
    if (ref.title) return ref.title;
    if (ref.filename) return ref.filename;
    return ref.tool || '참고자료';
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

declare global {
    interface Window {
        TypeHangul: any;
        Hangul: any;
        Typed: any;
    }
}

const ThinkingMessage = memo(() => {
    const [thinkingMessage, setThinkingMessage] = useState("");

    const thinkingMessages = [
        "한 번에 한 가지 질문을 하면 더 좋은 답변을 드릴 수 있어요!",
        "정확한 기업명을 넣어 질문해주시면 더 좋은 답변을 드릴 수 있어요!",
        "재무보고서, 애널리스트 보고서, 웹 중 어떤 정보를 참고할지 지정해주시면 거기서 찾아올게요!",
        "알고 계셨나요? 흐린 날에는 주식 수익률이 하락한다는 연구가 있습니다!",
        "주식 시장은 인내심이 없는 자로부터 인내심이 많은 자에게로 돈이 넘어가도록 설계되어 있다  - 워렌 버핏",
        "저축과 투자의 첫 번째 목표는 인플레이션을 이기는 것이다. 여러분의 돈은 거꾸로 돌아가는 쳇바퀴에 있다.   - 피터 린치",
        "개를 데리고 산책을 나갈 때 개가 주인보다 앞서갈 수는 있어도 주인을 떠날 수는 없다. 여기서 개는 주식 가격이고, 주인은 기업가치이다.  - 앙드레 코스톨라니",
        "현명한 투자자는 비관주의자에게 주식을 사서 낙관주의자에게 판다. - 벤자민 그레이엄",
        "저희가 제공하는 출처를 답변과 함께 보시면 더 정확한 정보를 얻으실 수 있습니다.",
        "알고 계셨나요? DONI를 테스트하기 위해 만든 100여 개의 질문-정답 쌍은 전부 사람이 직접 만들었습니다.",
        "알고 계셨나요? 어떤 기업은 매출이란 표현 대신 영업수익이란 표현을 사용합니다.",
        "DONI는 KOSPI기업에 대한 검색에 최적화되어 있습니다.",
        "알고 계셨나요? 네이버 인근에는 '동봉관'이라는 엄청난 맛집이 있습니다. 팀원 중 누군가는 프로젝트 기간 중 12회 방문했습니다.",
        "SNU 9기 파이팅 ♥ ♥",
        "알고 계셨나요? 기업이 자사주를 매입하면 주가가 상승하는 경향이 있지만, 장기적으로는 반드시 긍정적인 영향을 미치지는 않습니다.",
        "알고 계셨나요? 미국에서는 IPO 첫날 주가가 크게 상승하는 것을 'IPO 팝(Pop)'이라고 부릅니다.",
        "알고 계셨나요? 워런 버핏은 투자 결정을 내릴 때 '내가 이 기업의 전부를 산다면?'이라는 질문을 항상 먼저 한다고 합니다.",
        "알고 계셨나요? 달러 강세는 종종 해외 매출 비중이 높은 기업에 부정적인 영향을 미칠 수 있습니다.",
        "알고 계셨나요? 주식 리딩방은 투자자를 속이는 사기 수단으로 악용되는 경우가 많습니다."
    ];

    useEffect(() => {
        const interval = setInterval(() => {
            const randomIndex = Math.floor(Math.random() * thinkingMessages.length);
            setThinkingMessage(thinkingMessages[randomIndex]);
        }, 2700);
        
        const initialIndex = Math.floor(Math.random() * thinkingMessages.length);
        setThinkingMessage(thinkingMessages[initialIndex]);
        
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="message-container">
            <div className="avatar">
                <img src="/component/imgs/robot_avatar.png" alt="AI 챗봇" />
            </div>
            <div className="message bot-message thinking">
                <div className="thinking-header">
                    생각 중
                    <div className="loading-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
                <div className="thinking-tip">
                    Tip: {thinkingMessage}
                </div>
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

const PlanningPage = React.forwardRef<HTMLDivElement, { plans: PlanStep[] }>((props, ref) => {
    return (
        <div className="page planning-page" ref={ref}>
            <div className="page-header">
                <h2>🤔 생각의 과정</h2>
            </div>
            <div className="planning-container">
                {props.plans.map((plan, idx) => (
                    <div key={idx} className={`plan-step ${plan.status}`}>
                        <div className="plan-number">{idx + 1}</div>
                        <div className="plan-content">
                            <div className="plan-tool">{plan.tool}</div>
                            <div className="plan-description">{plan.description}</div>
                        </div>
                        <div className="plan-status">
                            {plan.status === 'pending' && '⏳'}
                            {plan.status === 'running' && '🔄'}
                            {plan.status === 'completed' && '✅'}
                        </div>
                    </div>
                ))}
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

// 도구 타입별 표시 이름 매핑
const toolTypeIcons = {
    'combined_financial_report_search': '📑 재무제표 검색',
    'web_search': '🌐 웹 검색',
    'math': '🔢 수식 계산',
    'report_analysis': '📊 리포트 분석',
    'stock_analysis': '📈 주가 분석',
    'combined_analysis': '🔍 종합 분석',
    'sector_analysis': '🏢 섹터 분석',
    'market_data': '📊 시장 데이터',
    'join': '🔗 데이터 결합',
    '기타': '🔧 기타'
};

const TaskProgressDisplay = memo(({ taskEvents }: { taskEvents: TaskProgress[] }) => {
    return (
        <div className="task-progress-container">
            <div className="task-events-list">
                {taskEvents.map((event, index) => (
                    <div key={`${event.timestamp}-${index}`} className={`task-event ${event.status}`}>
                        <div className="task-event-header">
                            <span className="task-type">
                                {toolTypeIcons[event.tool_type as keyof typeof toolTypeIcons] || 
                                 toolTypeIcons[event.task_name as keyof typeof toolTypeIcons]}
                            </span>
                            <span className={`task-status ${event.status}`}>
                                {event.status === 'running' && '실행 중'}
                                {event.status === 'completed' && '완료'}
                                {event.status === 'error' && '오류'}
                                {event.status === 'pending' && '대기 중'}
                            </span>
                        </div>
                        <div className="task-timestamp">
                            {new Date(event.timestamp).toLocaleTimeString()}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
});

function App() {
    const [messages, setMessages] = useState<Message[]>([{
        content: "안녕하세요! 궁금한 정보를 자유롭게 질문해주세요.",
        isUser: false,
        timestamp: new Date().toLocaleTimeString()
    }]);
    const [isThinking, setIsThinking] = useState(false);
    const [references, setReferences] = useState<Reference[]>([]);
    const [planSteps, setPlanSteps] = useState<PlanStep[]>([]);
    const [currentQuery, setCurrentQuery] = useState<string>("");
    const [taskProgress, setTaskProgress] = useState<TaskProgress[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const wsRef = useRef<WebSocket | null>(null);
    const [debugMessages, setDebugMessages] = useState<string[]>([]);
    const [isFading, setIsFading] = useState(false);
    const taskQueueRef = useRef<TaskProgress[]>([]);
    const [taskEvents, setTaskEvents] = useState<TaskProgress[]>([]);
    const [isDebugVisible, setIsDebugVisible] = useState(false);

    // 태스크 큐에 새로운 태스크 추가
    const addToTaskQueue = useCallback((task: TaskProgress) => {
        // task_id가 있는 경우에만 추가
        if (task.task_id || task.type === 'execution' || task.type === 'execution_summary') {
            taskQueueRef.current.push({
                ...task,
                tool_type: task.tool_type || task.task_name || '기타',
                timestamp: task.timestamp || new Date().toISOString()
            });
        }
    }, []);

    // 주기적으로 태스크 상태 업데이트 (100ms 간격)
    useEffect(() => {
        const updateInterval = setInterval(() => {
            if (taskQueueRef.current.length > 0) {
                setTaskProgress(prev => {
                    const newTasks = [...prev];
                    
                    taskQueueRef.current.forEach(task => {
                        // task_id가 있거나 execution 타입인 경우만 처리
                        if (task.task_id || task.type === 'execution') {
                            const existingIndex = newTasks.findIndex(t => t.task_id === task.task_id);
                            if (existingIndex !== -1) {
                                newTasks[existingIndex] = { 
                                    ...newTasks[existingIndex], 
                                    ...task,
                                    status: task.status || newTasks[existingIndex].status
                                };
                            } else {
                                newTasks.push(task);
                            }
                        }
                    });
                    
                    // 상태별로 정렬
                    const sortedTasks = newTasks.sort((a, b) => {
                        if (a.task_id && b.task_id) {
                            return parseInt(a.task_id) - parseInt(b.task_id);
                        }
                        return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
                    });
                    
                    taskQueueRef.current = []; // 큐 초기화
                    return sortedTasks;
                });
            }
        }, 100);

        return () => clearInterval(updateInterval);
    }, []);

    const clearPreviousChat = useCallback(() => {
        setIsFading(true);
        // 즉시 초기화
        setReferences([]);
        setPlanSteps([]);
        setTaskProgress([]);
        setDebugMessages([]);
        taskQueueRef.current = []; // 태스크 큐도 초기화
        
        // 페이드아웃 효과 후 페이드인
        setTimeout(() => {
            setIsFading(false);
        }, 500);
    }, []);

    const handleSubmit = async (query: string) => {
        if (!query.trim()) return;
        
        // 새로운 질문이 들어오면 모든 데이터 초기화
        clearPreviousChat();
        setCurrentQuery(query);
        // 작업 진행 상황과 디버그 콘솔 초기화
        setTaskEvents([]);
        setDebugMessages([]);
        setTaskProgress([]);
        taskQueueRef.current = [];

        const userMessage: Message = {
            content: query,
            isUser: true,
            timestamp: new Date().toLocaleTimeString()
        };

        setMessages(prev => [...prev, userMessage]);
        setIsThinking(true);

        try {
            const response = await fetch('http://localhost:8000/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query })
            });

            const data = await response.json();

            if (data.error) {
                const errorMessage: Message = {
                    content: data.message || "죄송합니다. 오류가 발생했습니다.",
                    isUser: false,
                    timestamp: new Date().toLocaleTimeString()
                };
                setMessages(prev => [...prev, errorMessage]);
            } else {
                const botMessage: Message = {
                    content: data.answer,
                    isUser: false,
                    timestamp: new Date().toLocaleTimeString()
                };
                setMessages(prev => [...prev, botMessage]);
                
                // 참고 자료가 있다면 업데이트
                if (data.docs && Array.isArray(data.docs)) {
                    setReferences(data.docs);
                }
            }
        } catch (error) {
            const errorMessage: Message = {
                content: "서버와의 통신 중 오류가 발생했습니다.",
                isUser: false,
                timestamp: new Date().toLocaleTimeString()
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsThinking(false);
        }
    };

    // 디버그 메시지 추가 함수
    const addDebugMessage = (message: string) => {
        setDebugMessages(prev => [...prev, `${new Date().toISOString()} - ${message}`]);
        console.log(message);
    };

    useEffect(() => {
        const connectWebSocket = () => {
            // 기존 웹소켓이 있다면 닫기
            if (wsRef.current) {
                wsRef.current.close();
            }

            const ws = new WebSocket('ws://localhost:8000/ws/task-progress');
            wsRef.current = ws;

            ws.onopen = () => {
                setIsConnected(true);
                addDebugMessage('WebSocket 연결됨');
            };

            ws.onmessage = (event) => {
                const data: TaskEvent = JSON.parse(event.data);
                addDebugMessage(`새로운 태스크 이벤트 수신: ${JSON.stringify(data)}`);
                
                if (data.type === 'task_progress') {
                    const taskData = data.data;
                    
                    // 기타 타입이면 무시
                    if (taskData.tool_type === '기타' || taskData.task_name === '기타') {
                        return;
                    }
                    
                    // taskEvents 상태 업데이트
                    setTaskEvents(prev => {
                        // 동일한 task_id와 tool_type을 가진 이벤트 찾기
                        const existingEventIndex = prev.findIndex(
                            event => event.task_id === taskData.task_id && 
                                    event.tool_type === taskData.tool_type
                        );

                        // 이미 존재하는 이벤트라면
                        if (existingEventIndex !== -1) {
                            // 상태가 같으면 무시
                            if (prev[existingEventIndex].status === taskData.status) {
                                return prev;
                            }
                            
                            // 상태가 다르면 업데이트
                            const newEvents = [...prev];
                            newEvents[existingEventIndex] = {
                                ...newEvents[existingEventIndex],
                                status: taskData.status,
                                timestamp: taskData.timestamp
                            };
                            return newEvents;
                        }
                        
                        // 새로운 이벤트라면 추가
                        return [...prev, taskData];
                    });
                }
            };

            ws.onerror = (error) => {
                addDebugMessage(`WebSocket 오류: ${error}`);
                setIsConnected(false);
            };

            ws.onclose = () => {
                setIsConnected(false);
                addDebugMessage('WebSocket 연결 종료');
                // 재연결 시도
                setTimeout(connectWebSocket, 3000);
            };
        };

        connectWebSocket();

        // 컴포넌트 언마운트 시 웹소켓 연결 종료
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [currentQuery]); // currentQuery가 변경될 때마다 웹소켓 재연결

    return (
        <div className="app-container">
            <div className="sections-container">
                <section className="task-progress-section">
                    <div className="section-header">
                        <h2>🔄 진행 상황</h2>
                    </div>
                    <TaskProgressDisplay taskEvents={taskEvents} />
                </section>

                <section className="chat-section">
                    <div className="section-header">
                        <h2>💬 대화</h2>
                    </div>
                    <ChatMessages messages={messages} isThinking={isThinking} />
                </section>

                <section className="reference-section">
                    <div className="section-header">
                        <div className="header-with-button">
                            <h2>📚 참고 자료</h2>
                            <button 
                                className="debug-toggle-button"
                                onClick={() => setIsDebugVisible(!isDebugVisible)}
                            >
                                {isDebugVisible ? '🔽 디버그 숨기기' : '🔼 디버그 보기'}
                            </button>
                        </div>
                    </div>
                    <div className="reference-content-wrapper">
                        <div className={`references-container ${isFading || isThinking ? 'fade-out' : 'fade-in'}`}>
                            {references.map((ref, idx) => (
                                <div key={idx} className="reference-item">
                                    <div className="reference-title">
                                        {typeIcons[getDisplayType(ref) as keyof typeof typeIcons]} {getDisplayTitle(ref)}
                                        {ref.page_number && ` (p.${ref.page_number})`}
                                    </div>
                                    {ref.broker && <div className="reference-details">{ref.broker}</div>}
                                    <div className="reference-content">{ref.referenced_content || ref.content}</div>
                                    {ref.link && (
                                        <a href={ref.link} target="_blank" rel="noopener noreferrer" className="reference-link">
                                            원문 보기
                                        </a>
                                    )}
                                </div>
                            ))}
                        </div>
                        {isDebugVisible && (
                            <div className="debug-console">
                                <h3>디버그 콘솔</h3>
                                <div className="debug-messages">
                                    {debugMessages.map((message, index) => (
                                        <div key={index} className="debug-message">
                                            {message}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </section>
            </div>
            
            <InputComponent onSend={handleSubmit} isThinking={isThinking} />
        </div>
    );
}

export default App; 