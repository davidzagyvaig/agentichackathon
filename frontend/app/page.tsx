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
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  isThinking?: boolean;
}

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
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<any[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize welcome message on client side only to avoid hydration mismatch
  useEffect(() => {
    fetchConversations();
    // Default welcome message
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content: "Hello! I'm ClaimGraph. I don't summarize papers; I **verify** their claims against ground truth. \n\nAsk me about a scientific topic (e.g., 'Do transformer models effectively predict climate tipping points?'), and I'll build a verification graph for you.",
        timestamp: new Date(),
      },
    ]);
  }, []);

  const fetchConversations = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/conversations");
      const data = await res.json();
      if (Array.isArray(data)) {
        setConversations(data);
      }
    } catch (e) {
      console.error("Failed to fetch conversations", e);
    }
  };

  const createNewSession = async () => {
    setConversationId(null);
    setMessages([
      {
        id: "welcome-" + Date.now(),
        role: "assistant",
        content: "Starting a new verification session. What would you like to verify?",
        timestamp: new Date(),
      },
    ]);
    setNodes([]); // Clear graph
    setEdges([]);
    setGraphVisible(false);
  };

  const loadConversation = async (id: string) => {
    setConversationId(id);
    setGraphVisible(false); // Maybe keep it handled by user?
    try {
      const res = await fetch(`http://localhost:8000/api/conversations/${id}`);
      const data = await res.json();
      // Transform backend messages to frontend format
      const loadedMessages: Message[] = data.map((msg: any) => ({
        id: msg.id,
        role: msg.role,
        content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
        timestamp: new Date(msg.created_at)
      }));
      setMessages(loadedMessages);
    } catch (e) {
      console.error("Failed to load conversation", e);
    }
  };

  // Graph State
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [graphVisible, setGraphVisible] = useState(false);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    // Create session if not exists
    let currentConvId = conversationId;
    if (!currentConvId) {
      try {
        const res = await fetch(`http://localhost:8000/api/conversations?title=${encodeURIComponent(inputValue.substring(0, 30))}`, { method: 'POST' });
        const newConv = await res.json();
        currentConvId = newConv.id;
        setConversationId(currentConvId);
        fetchConversations(); // Refresh sidebar
      } catch (e) {
        console.error("Failed to create conversation", e);
      }
    }

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue("");
    setIsAnalyzing(true);

    // Add thinking message
    const thinkingId = "thinking-" + Date.now();
    setMessages((prev) => [
      ...prev,
      {
        id: thinkingId,
        role: "assistant",
        content: "🔍 Searching literature, extracting claims, and verifying citations...",
        timestamp: new Date(),
        isThinking: true,
      },
    ]);

    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userMsg.content,
          max_papers: 5,
          extract_claims: true,
          validate_citations: true,
          conversation_id: currentConvId // Pass ID for persistence
        }),
      });

      const data = await response.json();

      // Remove thinking message
      setMessages((prev) => prev.filter((m) => m.id !== thinkingId));

      // Add response message
      const responseMsg: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: `I've analyzed ${data.summary.papers_analyzed} papers and extracted ${data.summary.claims_extracted} claims.\n\n` +
          `Found **${data.summary.claims_verified} verified claims** and **${data.summary.claims_suspicious} suspicious items**.\n\n` +
          `I've built a knowledge graph on the right. You can now deeply verify citations by clicking on citation nodes.`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, responseMsg]);

      // Update Graph by fetching from backend
      await fetchGraph();
      setGraphVisible(true);

    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => prev.filter((m) => m.id !== thinkingId));
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: "Sorry, I encountered an error analyzing that topic. Please try again.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const fetchGraph = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/graph?format=reactflow");
      const data = await response.json();

      if (data.nodes && data.edges) {
        setNodes(data.nodes);
        setEdges(data.edges);
      }
    } catch (error) {
      console.error("Failed to fetch graph:", error);
    }
  };

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    // Show details in chat or modal
    if (node.id.startsWith("paper-")) {
      const paper = node.data;
      if (paper) setSelectedNode({ type: 'paper', data: paper });
    } else if (node.id.startsWith("claim-")) {
      const claim = node.data;
      if (claim) setSelectedNode({ type: 'claim', data: claim });
    } else if (node.id.startsWith("citation-")) {
      const citation = node.data;
      if (citation) setSelectedNode({ type: 'citation', data: citation });
    }
    setGraphVisible(true);
  }, []);

  const handleDeepVerify = async (citation: any) => {
    // Trigger analysis for this citation
    if (!citation.title) return;

    setGraphVisible(true); // Keep visible? Or switch to chat? Switch to chat to see progress.
    // But we are in split view on desktop.

    const newQuery = `Verify paper: ${citation.title}`;
    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: newQuery,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsAnalyzing(true);

    // Close details
    setSelectedNode(null);

    // Add thinking
    const thinkingId = "thinking-" + Date.now();
    setMessages((prev) => [
      ...prev,
      {
        id: thinkingId,
        role: "assistant",
        content: "🔍 Deep verifying citation...",
        timestamp: new Date(),
        isThinking: true,
      },
    ]);

    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: citation.title, // Search by title
          max_papers: 1, // Focus on this one
          extract_claims: true,
          validate_citations: true,
          conversation_id: conversationId // Maintain context
        }),
      });
      const data = await response.json();
      // Remove thinking message
      setMessages((prev) => prev.filter((m) => m.id !== thinkingId));

      // Add response
      const responseMsg: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: `Deep verification complete for "${citation.title}".\n\nExtracted ${data.summary.claims_extracted} claims. Graph updated.`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, responseMsg]);

      await fetchGraph();

    } catch (error) {
      console.error(error);
      setMessages((prev) => prev.filter((m) => m.id !== thinkingId));
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="flex h-screen bg-black text-white overflow-hidden font-sans">
      {/* Sidebar - History */}
      <div className="w-[260px] bg-black hidden md:flex flex-col border-r border-gray-800">
        <div className="p-3">
          <button
            onClick={createNewSession}
            className="flex items-center gap-2 w-full px-3 py-2 text-sm text-gray-200 bg-gray-900 hover:bg-gray-800 rounded-md transition-colors border border-gray-700"
          >
            <span>+</span> New verification
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-3 py-2">
          <div className="text-xs font-semibold text-gray-500 mb-2 px-2">Recent</div>
          <div className="space-y-1">
            {conversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => loadConversation(conv.id)}
                className={`w-full text-left px-3 py-2 text-sm rounded-md truncate transition-colors ${conversationId === conv.id ? 'bg-gray-800 text-white' : 'text-gray-300 hover:bg-gray-900'}`}
              >
                {conv.title}
              </button>
            ))}
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
                <div className={`max-w-[85%] md:max-w-[80%] rounded-2xl px-5 py-3 text-sm leading-relaxed whitespace-pre-wrap ${msg.role === 'user'
                  ? 'bg-blue-600 text-white rounded-br-none'
                  : 'bg-gray-800 text-gray-100 rounded-bl-none border border-gray-700'
                  } ${msg.isThinking ? 'animate-pulse' : ''}`}>
                  {msg.content}
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
                placeholder="Ask a scientific question to verify..."
                className="w-full bg-gray-800 text-white rounded-xl pl-4 pr-12 py-3 md:py-4 focus:outline-none focus:ring-2 focus:ring-green-600/50 border border-gray-700 resize-none h-[60px] md:h-[100px] scrollbar-hide text-sm md:text-base"
              />
              <button
                onClick={handleSendMessage}
                disabled={isAnalyzing || !inputValue.trim()}
                className="absolute right-3 bottom-3 md:bottom-4 p-2 bg-green-600 text-white rounded-lg hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
              </button>
            </div>
            <div className="text-center mt-2">
              <p className="text-[10px] text-gray-500">ClaimGraph verifies claims against ground truth. AI can make mistakes.</p>
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
            <div className="flex gap-2">
              <span className="flex items-center gap-1 text-[10px] text-gray-400">
                <span className="w-2 h-2 rounded-full bg-green-500"></span> Verified
              </span>
              <span className="flex items-center gap-1 text-[10px] text-gray-400">
                <span className="w-2 h-2 rounded-full bg-red-500"></span> Suspicious
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
                    if (n.id.startsWith('paper')) return '#065f46';
                    if (n.id.startsWith('claim')) return '#374151';
                    if (n.id.startsWith('citation')) return '#4b5563';
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
                <p className="text-sm">Start a verification to build the graph</p>
              </div>
            )}

            {/* Details Overlay */}
            {selectedNode && (
              <div className="absolute top-4 right-4 w-80 bg-gray-900/95 backdrop-blur-md border border-gray-700 rounded-xl p-4 shadow-2xl overflow-y-auto max-h-[80%]">
                <div className="flex justify-between items-start mb-3">
                  <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">
                    {selectedNode.type === 'paper' ? 'Paper Details' :
                      selectedNode.type === 'claim' ? 'Claim Verification' : 'Citation Source'}
                  </span>
                  <button onClick={() => setSelectedNode(null)} className="text-gray-400 hover:text-white">✕</button>
                </div>

                {selectedNode.type === 'paper' && (
                  <div className="space-y-3">
                    <h3 className="font-bold text-white leading-snug">{selectedNode.data.title}</h3>
                    <div className="text-xs text-gray-400">
                      {selectedNode.data.authors?.join(', ')} • {selectedNode.data.year}
                    </div>
                    <div className="p-3 bg-gray-800 rounded-lg text-xs text-gray-300 leading-relaxed max-h-40 overflow-y-auto">
                      {selectedNode.data.abstract}
                    </div>
                    <a
                      href={selectedNode.data.doi ? `https://doi.org/${selectedNode.data.doi}` : '#'}
                      target="_blank"
                      rel="noopener"
                      className="block text-center w-full py-2 bg-gray-800 hover:bg-gray-700 rounded text-xs text-green-400 font-semibold transition-colors"
                    >
                      View Full Paper ↗
                    </a>
                  </div>
                )}

                {selectedNode.type === 'claim' && (
                  <div className="space-y-3">
                    <div className={`p-2 rounded text-xs font-bold text-center uppercase tracking-wide
                                    ${selectedNode.data.validation_status === 'verified' ? 'bg-green-900/50 text-green-400 border border-green-900' :
                        selectedNode.data.validation_status === 'suspicious' ? 'bg-red-900/50 text-red-400 border border-red-900' :
                          'bg-gray-800 text-gray-400'}`}>
                      {selectedNode.data.validation_status}
                    </div>

                    <p className="text-sm text-white leading-relaxed font-medium">
                      &quot;{selectedNode.data.text}&quot;
                    </p>

                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="p-2 bg-gray-800 rounded">
                        <div className="text-gray-500 mb-1">Type</div>
                        <div className="text-gray-300 capitalize">{selectedNode.data.claim_type}</div>
                      </div>
                      <div className="p-2 bg-gray-800 rounded">
                        <div className="text-gray-500 mb-1">Evidence</div>
                        <div className="text-gray-300 capitalize">{selectedNode.data.evidence_type}</div>
                      </div>
                    </div>
                  </div>
                )}

                {selectedNode.type === 'citation' && (
                  <div className="space-y-3">
                    <h3 className="font-bold text-white leading-snug">{selectedNode.data.title}</h3>
                    <div className="flex gap-2">
                      <div className={`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide
                                    ${selectedNode.data.is_retracted ? 'bg-red-900/50 text-red-400' : 'bg-gray-800 text-gray-400'}`}>
                        {selectedNode.data.is_retracted ? 'RETRACTED' : 'CITATION'}
                      </div>
                      {selectedNode.data.is_relevant ? (
                        <div className="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide bg-green-900/30 text-green-400">
                          RELEVANT
                        </div>
                      ) : (
                        <div className="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide bg-gray-800 text-gray-500">
                          NOT RELEVANT
                        </div>
                      )}
                    </div>

                    {selectedNode.data.validation_notes && (
                      <div className="p-3 bg-gray-800 rounded-lg text-xs text-gray-300 leading-relaxed italic border-l-2 border-green-700">
                        &quot;{selectedNode.data.validation_notes}&quot;
                      </div>
                    )}

                    <a
                      href={selectedNode.data.doi ? `https://doi.org/${selectedNode.data.doi}` : '#'}
                      target="_blank"
                      rel="noopener"
                      className="block text-center w-full py-2 bg-gray-800 hover:bg-gray-700 rounded text-xs text-blue-400 font-semibold transition-colors disabled:opacity-50"
                    >
                      View Source ↗
                    </a>

                    <button
                      onClick={() => handleDeepVerify(selectedNode.data)}
                      className="w-full py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-lg text-xs font-bold transition-all shadow-lg flex items-center justify-center gap-2 mt-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"></path></svg>
                      Verify This Citation (Deep Dive)
                    </button>
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
