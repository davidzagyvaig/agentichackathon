"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  BackgroundVariant,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// Types
interface ClarificationQuestion {
  question: string;
  options: string[];
  required: boolean;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  isThinking?: boolean;
  clarifications?: ClarificationQuestion[];
  isDeepResearchReport?: boolean;
  contradictions?: { summary: string }[];
}

// Research stages for progress indicator
const DEEP_RESEARCH_STAGES = [
  { id: 'clarify', label: 'Understanding your question...' },
  { id: 'enrich', label: 'Planning research strategy...' },
  { id: 'research', label: 'Deep research in progress...' },
  { id: 'complete', label: 'Compiling report...' },
];

const STATUS_COLORS = {
  verified: "#22c55e",
  suspicious: "#ef4444",
  pending: "#6b7280",
};

export default function Home() {
  // Chat State
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [researchStage, setResearchStage] = useState<string | null>(null);
  const [pendingClarifications, setPendingClarifications] = useState<Record<string, string>>({});
  const [awaitingClarification, setAwaitingClarification] = useState(false);
  const [currentQuery, setCurrentQuery] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize welcome message on client side only to avoid hydration mismatch
  useEffect(() => {
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content: "Hello! I'm ClaimGraph Deep Research. I conduct comprehensive scientific analysis using a knowledge graph of verified claims.\n\nAsk me about a topic (e.g., 'What is the relationship between gut microbiome and depression?'), and I'll analyze the evidence, identify contradictions, and provide a grounded research report.",
        timestamp: new Date(),
      },
    ]);
  }, []);

  // Graph State
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [graphVisible, setGraphVisible] = useState(false);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle clarification answer
  const handleClarificationAnswer = (question: string, answer: string) => {
    setPendingClarifications(prev => ({
      ...prev,
      [question]: answer
    }));
  };

  // Submit clarifications and continue research
  const submitClarifications = async () => {
    setAwaitingClarification(false);
    
    // Add user's clarification answers as a message
    const answersText = Object.entries(pendingClarifications)
      .map(([q, a]) => `${q}: ${a}`)
      .join('\n');
    
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      role: "user",
      content: answersText,
      timestamp: new Date(),
    }]);

    // Continue with deep research
    await executeDeepResearch(currentQuery, pendingClarifications);
    setPendingClarifications({});
  };

  // Main deep research flow
  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userQuery = inputValue;
    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: userQuery,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue("");
    setIsAnalyzing(true);
    setCurrentQuery(userQuery);
    setResearchStage('clarify');

    try {
      // Step 1: Check if clarification is needed
      const clarifyRes = await fetch('http://localhost:8000/api/deep-research/clarify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery }),
      });
      const clarifyData = await clarifyRes.json();

      if (clarifyData.needs_clarification && clarifyData.questions?.length > 0) {
        // Show clarification questions
        setResearchStage(null);
        setAwaitingClarification(true);
        setMessages(prev => [...prev, {
          id: 'clarify-' + Date.now(),
          role: 'assistant',
          content: 'I have a few questions to better understand your research needs:',
          clarifications: clarifyData.questions,
          timestamp: new Date(),
        }]);
        setIsAnalyzing(false);
        return;
      }

      // No clarification needed, proceed with research
      await executeDeepResearch(userQuery, {});

    } catch (error) {
      console.error("Error:", error);
      setResearchStage(null);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again.",
        timestamp: new Date(),
      }]);
    } finally {
      if (!awaitingClarification) {
        setIsAnalyzing(false);
        setResearchStage(null);
      }
    }
  };

  // Execute deep research with optional clarifications
  const executeDeepResearch = async (query: string, clarifications: Record<string, string>) => {
    setIsAnalyzing(true);
    setResearchStage('enrich');

    // Add progress message
    const progressId = 'progress-' + Date.now();
    setMessages(prev => [...prev, {
      id: progressId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isThinking: true,
    }]);

    try {
      setResearchStage('research');

      const response = await fetch('http://localhost:8000/api/deep-research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          clarifications,
          max_tool_calls: 50,
          use_background: false, // Use sync for faster demo
        }),
      });

      setResearchStage('complete');
      const data = await response.json();

      // Remove progress message
      setMessages(prev => prev.filter(m => m.id !== progressId));

      // Add research report
      const reportMsg: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: data.report || 'Research completed but no report was generated.',
        timestamp: new Date(),
        isDeepResearchReport: true,
        contradictions: data.contradictions || [],
      };
      setMessages(prev => [...prev, reportMsg]);

      // Fetch and display the mock graph
      await fetchMockGraph();
      setGraphVisible(true);

    } catch (error) {
      console.error("Deep research error:", error);
      setMessages(prev => prev.filter(m => m.id !== progressId));
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'assistant',
        content: 'Sorry, the deep research encountered an error. Please try again.',
        timestamp: new Date(),
      }]);
    } finally {
      setIsAnalyzing(false);
      setResearchStage(null);
    }
  };

  // Fetch mock graph for visualization
  const fetchMockGraph = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/deep-research/graph");
      const data = await response.json();

      if (data.nodes && data.edges) {
        // Convert mock graph format to React Flow format
        const flowNodes: Node[] = data.nodes.map((n: any, i: number) => ({
          id: n.id,
          position: { 
            x: n.type === 'paper' ? 100 : 400, 
            y: i * 80 
          },
          data: { 
            label: n.label,
            ...n.metadata,
            type: n.type
          },
          style: {
            background: n.type === 'paper' ? '#065f46' : '#374151',
            color: 'white',
            border: '1px solid #4b5563',
            borderRadius: '8px',
            padding: '10px',
            fontSize: '12px',
            width: 200,
          }
        }));

        const flowEdges: Edge[] = data.edges.map((e: any) => ({
          id: e.id,
          source: e.source,
          target: e.target,
          label: e.type,
          style: {
            stroke: e.type === 'contradicts' ? '#ef4444' : e.type === 'supports' ? '#22c55e' : '#6b7280',
          },
          labelStyle: { fill: '#9ca3af', fontSize: 10 },
          animated: e.type === 'contradicts',
        }));

        setNodes(flowNodes);
        setEdges(flowEdges);
      }
    } catch (error) {
      console.error("Failed to fetch graph:", error);
    }
  };

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode({ type: node.data.type || 'paper', data: node.data });
    setGraphVisible(true);
  }, []);

  // Format report text with clickable citations
  const formatReportContent = (text: string) => {
    // Replace [node_id] patterns with styled spans
    return text.replace(/\[([^\]]+)\]/g, (match, nodeId) => {
      if (nodeId.startsWith('paper_') || nodeId.startsWith('claim_')) {
        return `<span class="text-green-400 cursor-pointer hover:underline" data-node="${nodeId}">[${nodeId}]</span>`;
      }
      return match;
    });
  };

  return (
    <div className="flex h-screen bg-black text-white overflow-hidden font-sans">
      {/* Sidebar - History */}
      <div className="w-[260px] bg-black hidden md:flex flex-col border-r border-gray-800">
        <div className="p-3">
          <button
            onClick={() => {
              setGraphVisible(false);
              setMessages([]);
              setAwaitingClarification(false);
              setPendingClarifications({});
              setResearchStage(null);
              setMessages([
                {
                  id: "welcome",
                  role: "assistant",
                  content: "Hello! I'm ClaimGraph Deep Research. I conduct comprehensive scientific analysis using a knowledge graph of verified claims.\n\nAsk me about a topic (e.g., 'What is the relationship between gut microbiome and depression?'), and I'll analyze the evidence, identify contradictions, and provide a grounded research report.",
                  timestamp: new Date(),
                },
              ]);
            }}
            className="flex items-center gap-2 w-full px-3 py-2 text-sm text-gray-200 bg-gray-900 hover:bg-gray-800 rounded-md transition-colors border border-gray-700"
          >
            <span>+</span> New Research
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-3 py-2">
          <div className="text-xs font-semibold text-gray-500 mb-2 px-2">Recent</div>
          <div className="space-y-1">
            <button className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-gray-900 rounded-md truncate transition-colors">
              Gut microbiome depression...
            </button>
            <button className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-gray-900 rounded-md truncate transition-colors">
              Probiotic effectiveness...
            </button>
          </div>
        </div>
        <div className="p-4 border-t border-gray-800">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <div className="w-6 h-6 rounded-full bg-green-900 flex items-center justify-center text-green-500 font-bold text-xs">U</div>
            User Account
          </div>
        </div>
      </div>

      {/* Main Content Split */}
      <div className="flex-1 flex flex-col md:flex-row h-full relative">

        {/* Chat Interface */}
        <div className={`flex-1 flex flex-col h-full bg-[#0a0a0a] transition-all duration-300 ${graphVisible ? 'md:w-[45%]' : 'w-full'} border-r border-gray-800`}>
          {/* Top Bar for Mobile */}
          <div className="md:hidden flex items-center justify-between p-4 border-b border-gray-800">
            <span className="font-bold">ClaimGraph</span>
            <button className="text-gray-400" onClick={() => setGraphVisible(!graphVisible)}>
              {graphVisible ? 'Show Chat' : 'Show Graph'}
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 scrollbar-thin scrollbar-thumb-gray-800">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-full bg-green-600 flex-shrink-0 flex items-center justify-center text-white font-bold text-xs">
                    CG
                  </div>
                )}
                <div className={`max-w-[85%] md:max-w-[80%] rounded-2xl px-5 py-3 text-sm leading-relaxed ${msg.role === 'user'
                  ? 'bg-blue-600 text-white rounded-br-none'
                  : 'bg-gray-800 text-gray-100 rounded-bl-none border border-gray-700'
                  } ${msg.isThinking ? 'animate-pulse' : ''}`}>
                  
                  {/* Progress indicator for thinking state */}
                  {msg.isThinking && researchStage && (
                    <div className="flex items-center gap-2 p-2 bg-green-900/30 rounded-lg border border-green-700">
                      <div className="animate-spin w-4 h-4 border-2 border-green-500 border-t-transparent rounded-full" />
                      <span className="text-sm text-green-300">
                        {DEEP_RESEARCH_STAGES.find(s => s.id === researchStage)?.label}
                      </span>
                    </div>
                  )}

                  {/* Regular message content */}
                  {!msg.isThinking && !msg.isDeepResearchReport && (
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                  )}

                  {/* Deep research report */}
                  {msg.isDeepResearchReport && (
                    <div className="space-y-4">
                      <div className="p-4 bg-green-900/20 rounded-lg border border-green-700">
                        <h4 className="text-green-300 font-semibold mb-2 flex items-center gap-2">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          Research Report
                        </h4>
                        <div 
                          className="whitespace-pre-wrap text-gray-200"
                          dangerouslySetInnerHTML={{ __html: formatReportContent(msg.content) }}
                        />
                      </div>
                      
                      {msg.contradictions && msg.contradictions.length > 0 && (
                        <div className="p-3 bg-red-900/20 rounded-lg border border-red-700">
                          <h5 className="text-red-300 font-semibold mb-2 flex items-center gap-2">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                            Contradictions Found
                          </h5>
                          {msg.contradictions.map((c, i) => (
                            <div key={i} className="text-sm text-red-200 py-1">{c.summary}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Clarification questions */}
                  {msg.clarifications && msg.clarifications.length > 0 && (
                    <div className="space-y-3 mt-3">
                      {msg.clarifications.map((q, i) => (
                        <div key={i} className="p-3 bg-gray-700/50 rounded-lg">
                          <p className="text-sm mb-2 text-gray-200">{q.question}</p>
                          <div className="flex flex-wrap gap-2">
                            {q.options && q.options.length > 0 ? (
                              q.options.map((opt) => (
                                <button
                                  key={opt}
                                  onClick={() => handleClarificationAnswer(q.question, opt)}
                                  className={`px-3 py-1 rounded text-xs transition-colors ${
                                    pendingClarifications[q.question] === opt
                                      ? 'bg-green-600 text-white'
                                      : 'bg-gray-600 hover:bg-gray-500 text-gray-200'
                                  }`}
                                >
                                  {opt}
                                </button>
                              ))
                            ) : (
                              <input
                                type="text"
                                placeholder="Type your answer..."
                                className="flex-1 px-3 py-1 bg-gray-600 rounded text-xs text-white placeholder-gray-400"
                                onChange={(e) => handleClarificationAnswer(q.question, e.target.value)}
                              />
                            )}
                          </div>
                        </div>
                      ))}
                      
                      {awaitingClarification && (
                        <button
                          onClick={submitClarifications}
                          disabled={Object.keys(pendingClarifications).length === 0}
                          className="w-full py-2 bg-green-600 hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-sm font-semibold transition-colors mt-2"
                        >
                          Continue Research →
                        </button>
                      )}
                    </div>
                  )}
                </div>
                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-gray-600 flex-shrink-0 flex items-center justify-center text-white font-bold text-xs">
                    U
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 md:p-6 bg-[#0a0a0a]">
            <div className="relative max-w-3xl mx-auto">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder="Ask a research question (e.g., 'What causes depression according to gut microbiome research?')"
                disabled={awaitingClarification}
                className="w-full bg-gray-800 text-white rounded-xl pl-4 pr-12 py-3 md:py-4 focus:outline-none focus:ring-2 focus:ring-green-600/50 border border-gray-700 resize-none h-[60px] md:h-[100px] scrollbar-hide text-sm md:text-base disabled:opacity-50"
              />
              <button
                onClick={handleSendMessage}
                disabled={isAnalyzing || !inputValue.trim() || awaitingClarification}
                className="absolute right-3 bottom-3 md:bottom-4 p-2 bg-green-600 text-white rounded-lg hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
              </button>
            </div>
            <div className="text-center mt-2">
              <p className="text-[10px] text-gray-500">ClaimGraph Deep Research - Powered by o3-deep-research. AI analysis may contain errors.</p>
            </div>
          </div>
        </div>

        {/* Graph Visualizer (Right Side) */}
        <div className={`hidden md:flex flex-col h-full bg-[#111111] transition-all duration-300 ${graphVisible ? 'w-[55%]' : 'w-0 opacity-0 overflow-hidden'}`}>
          <div className="h-14 border-b border-gray-800 flex items-center justify-between px-4 bg-gray-900/50">
            <span className="font-semibold text-gray-200 text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500"></span>
              Knowledge Graph
            </span>
            <div className="flex gap-3">
              <span className="flex items-center gap-1 text-[10px] text-gray-400">
                <span className="w-2 h-2 rounded-full bg-green-500"></span> Supports
              </span>
              <span className="flex items-center gap-1 text-[10px] text-gray-400">
                <span className="w-2 h-2 rounded-full bg-red-500"></span> Contradicts
              </span>
              <span className="flex items-center gap-1 text-[10px] text-gray-400">
                <span className="w-2 h-2 rounded-full bg-gray-500"></span> Cites
              </span>
            </div>
          </div>

          <div className="flex-1 relative">
            {nodes.length > 0 ? (
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onNodeClick={onNodeClick}
                fitView
                className="bg-[#111111]"
              >
                <Background color="#333" gap={20} variant={BackgroundVariant.Dots} />
                <Controls className="bg-gray-800 border-gray-700 fill-gray-200" />
                <MiniMap
                  className="bg-gray-800 border-gray-700"
                  nodeColor={(n) => {
                    if (n.data?.type === 'paper') return '#065f46';
                    if (n.data?.type === 'claim') return '#374151';
                    return '#6b7280';
                  }}
                />
              </ReactFlow>
            ) : (
              <div className="flex h-full items-center justify-center text-gray-500 flex-col gap-2">
                <svg className="w-12 h-12 opacity-20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="2" y1="12" x2="22" y2="12"></line>
                  <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                </svg>
                <p className="text-sm">Start a research query to see the knowledge graph</p>
              </div>
            )}

            {/* Details Overlay */}
            {selectedNode && (
              <div className="absolute top-4 right-4 w-80 bg-gray-900/95 backdrop-blur-md border border-gray-700 rounded-xl p-4 shadow-2xl overflow-y-auto max-h-[80%]">
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">
                    {selectedNode.type === 'paper' ? 'Paper Details' : 'Claim Details'}
                  </span>
                  <button onClick={() => setSelectedNode(null)} className="text-gray-400 hover:text-white">✕</button>
                </div>

                {selectedNode.type === 'paper' && (
                  <div className="space-y-3">
                    <h3 className="font-bold text-white leading-snug">{selectedNode.data.label || selectedNode.data.title}</h3>
                    {selectedNode.data.authors && (
                      <div className="text-xs text-gray-400">
                        {Array.isArray(selectedNode.data.authors) ? selectedNode.data.authors.join(', ') : selectedNode.data.authors} • {selectedNode.data.year}
                      </div>
                    )}
                    {selectedNode.data.abstract && (
                      <div className="p-3 bg-gray-800 rounded-lg text-xs text-gray-300 leading-relaxed max-h-40 overflow-y-auto">
                        {selectedNode.data.abstract}
                      </div>
                    )}
                    {selectedNode.data.citations && (
                      <div className="text-xs text-gray-400">
                        Citations: {selectedNode.data.citations}
                      </div>
                    )}
                  </div>
                )}

                {selectedNode.type === 'claim' && (
                  <div className="space-y-3">
                    <p className="text-sm text-white leading-relaxed">
                      {selectedNode.data.label || selectedNode.data.description}
                    </p>
                    {selectedNode.data.confidence && (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-400">Confidence:</span>
                        <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-green-500 rounded-full"
                            style={{ width: `${selectedNode.data.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-300">{Math.round(selectedNode.data.confidence * 100)}%</span>
                      </div>
                    )}
                    {selectedNode.data.consensus_level && (
                      <div className="text-xs text-gray-400">
                        Consensus: <span className="capitalize text-gray-300">{selectedNode.data.consensus_level}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
