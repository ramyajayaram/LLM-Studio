/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles, Trash2, Loader2, Brain, Play, Code, Terminal, Cpu } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import Markdown from 'react-markdown';
import { sendMessageStream, Message } from './services/geminiService';
import { Tokenizer } from './lib/tokenizer';
import { ToyGPT } from './lib/transformer';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'playground'>('chat');
  
  // Chat State
  const [messages, setMessages] = useState<Message[]>([
    { role: 'model', content: 'Hello! I am Gemini. How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Playground State
  const [trainingText, setTrainingText] = useState("To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles, And by opposing end them?");
  const [isTraining, setIsTraining] = useState(false);
  const [loss, setLoss] = useState<number | null>(null);
  const [epoch, setEpoch] = useState(0);
  const [generatedText, setGeneratedText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const modelRef = useRef<ToyGPT | null>(null);
  const tokenizerRef = useRef<Tokenizer | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (activeTab === 'chat') scrollToBottom();
  }, [messages, activeTab]);

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const assistantMessage: Message = { role: 'model', content: '' };
    setMessages(prev => [...prev, assistantMessage]);

    try {
      let fullResponse = '';
      const stream = sendMessageStream([...messages, userMessage]);
      
      for await (const chunk of stream) {
        fullResponse += chunk;
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = {
            role: 'model',
            content: fullResponse
          };
          return newMessages;
        });
      }
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const trainModel = async () => {
    if (isTraining) return;
    setIsTraining(true);
    setEpoch(0);
    setLoss(null);

    const tokenizer = new Tokenizer(trainingText);
    tokenizerRef.current = tokenizer;
    
    const model = new ToyGPT(tokenizer.vocabSize);
    modelRef.current = model;

    const encoded = tokenizer.encode(trainingText);
    
    await model.train(encoded, 50, (e, l) => {
      setEpoch(e + 1);
      setLoss(l);
    });

    setIsTraining(false);
  };

  const generateFromModel = async () => {
    if (!modelRef.current || !tokenizerRef.current || isGenerating) return;
    setIsGenerating(true);
    
    const seed = tokenizerRef.current.encode(trainingText.slice(0, 10));
    const resultIds = await modelRef.current.generate(seed, 100);
    const resultText = tokenizerRef.current.decode(resultIds);
    
    setGeneratedText(resultText);
    setIsGenerating(false);
  };

  return (
    <div className="flex flex-col h-screen bg-zinc-50 font-sans overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 bg-white border-b border-zinc-200 shadow-sm shrink-0">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-zinc-900 rounded-lg">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-lg font-semibold tracking-tight text-zinc-900">LLM Studio</h1>
          </div>
          
          <nav className="flex bg-zinc-100 p-1 rounded-xl ml-4">
            <button
              onClick={() => setActiveTab('chat')}
              className={cn(
                "px-4 py-1.5 text-sm font-medium rounded-lg transition-all",
                activeTab === 'chat' ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-500 hover:text-zinc-700"
              )}
            >
              Chat
            </button>
            <button
              onClick={() => setActiveTab('playground')}
              className={cn(
                "px-4 py-1.5 text-sm font-medium rounded-lg transition-all",
                activeTab === 'playground' ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-500 hover:text-zinc-700"
              )}
            >
              Model Playground
            </button>
          </nav>
        </div>
        
        {activeTab === 'chat' && (
          <button
            onClick={() => setMessages([{ role: 'model', content: 'Hello! I am Gemini. How can I help you today?' }])}
            className="p-2 text-zinc-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        )}
      </header>

      <main className="flex-1 overflow-hidden relative">
        <AnimatePresence mode="wait">
          {activeTab === 'chat' ? (
            <motion.div
              key="chat"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="h-full flex flex-col"
            >
              <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
                <div className="max-w-3xl mx-auto space-y-6">
                  {messages.map((msg, idx) => (
                    <div
                      key={idx}
                      className={cn(
                        "flex gap-4",
                        msg.role === 'user' ? "flex-row-reverse" : "flex-row"
                      )}
                    >
                      <div className={cn(
                        "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                        msg.role === 'user' ? "bg-zinc-900" : "bg-zinc-200"
                      )}>
                        {msg.role === 'user' ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-zinc-700" />}
                      </div>
                      <div className={cn(
                        "max-w-[85%] px-4 py-3 rounded-2xl shadow-sm",
                        msg.role === 'user' ? "bg-zinc-900 text-white rounded-tr-none" : "bg-white border border-zinc-200 text-zinc-800 rounded-tl-none"
                      )}>
                        <div className="markdown-body">
                          <Markdown>{msg.content || (isLoading && idx === messages.length - 1 ? '...' : '')}</Markdown>
                        </div>
                      </div>
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              </div>

              <footer className="p-4 md:p-6 bg-white border-t border-zinc-200">
                <div className="max-w-3xl mx-auto">
                  <form onSubmit={handleChatSubmit} className="relative">
                    <input
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="Ask Gemini anything..."
                      disabled={isLoading}
                      className="w-full pl-4 pr-12 py-4 bg-zinc-100 border-none rounded-2xl focus:ring-2 focus:ring-zinc-900 focus:bg-white transition-all outline-none text-zinc-900"
                    />
                    <button
                      type="submit"
                      disabled={isLoading || !input.trim()}
                      className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-zinc-900 text-white rounded-xl disabled:opacity-50 hover:bg-zinc-800 transition-colors"
                    >
                      <Send className="w-5 h-5" />
                    </button>
                  </form>
                </div>
              </footer>
            </motion.div>
          ) : (
            <motion.div
              key="playground"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="h-full overflow-y-auto p-6"
            >
              <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left Column: Training */}
                <div className="space-y-6">
                  <section className="bg-white p-6 rounded-2xl border border-zinc-200 shadow-sm">
                    <div className="flex items-center gap-2 mb-4">
                      <Code className="w-5 h-5 text-zinc-500" />
                      <h2 className="font-semibold text-zinc-900">1. Training Data</h2>
                    </div>
                    <textarea
                      value={trainingText}
                      onChange={(e) => setTrainingText(e.target.value)}
                      className="w-full h-32 p-4 bg-zinc-50 border border-zinc-200 rounded-xl font-mono text-sm resize-none outline-none focus:ring-2 focus:ring-zinc-900"
                      placeholder="Enter text to train the model on..."
                    />
                    <button
                      onClick={trainModel}
                      disabled={isTraining}
                      className="mt-4 w-full flex items-center justify-center gap-2 py-3 bg-zinc-900 text-white rounded-xl font-medium hover:bg-zinc-800 disabled:opacity-50 transition-all"
                    >
                      {isTraining ? <Loader2 className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
                      {isTraining ? `Training (Epoch ${epoch}/50)...` : "Train Toy GPT Model"}
                    </button>
                    
                    {loss !== null && (
                      <div className="mt-4 p-3 bg-zinc-50 rounded-lg border border-zinc-100 flex justify-between items-center">
                        <span className="text-xs text-zinc-500 uppercase tracking-wider font-bold">Current Loss</span>
                        <span className="font-mono text-zinc-900 font-semibold">{loss.toFixed(4)}</span>
                      </div>
                    )}
                  </section>

                  <section className="bg-zinc-900 text-zinc-400 p-6 rounded-2xl shadow-xl overflow-hidden relative">
                    <div className="flex items-center gap-2 mb-4 text-zinc-100">
                      <Cpu className="w-5 h-5" />
                      <h2 className="font-semibold">Model Architecture</h2>
                    </div>
                    <div className="space-y-2 font-mono text-xs">
                      <div className="flex justify-between border-b border-zinc-800 pb-1">
                        <span>Embedding</span>
                        <span className="text-emerald-400">64-dim</span>
                      </div>
                      <div className="flex justify-between border-b border-zinc-800 pb-1">
                        <span>Attention Heads</span>
                        <span className="text-emerald-400">4</span>
                      </div>
                      <div className="flex justify-between border-b border-zinc-800 pb-1">
                        <span>Transformer Blocks</span>
                        <span className="text-emerald-400">1</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Vocab Size</span>
                        <span className="text-emerald-400">{tokenizerRef.current?.vocabSize || 0}</span>
                      </div>
                    </div>
                    <div className="absolute -bottom-4 -right-4 opacity-10">
                      <Brain className="w-32 h-32" />
                    </div>
                  </section>
                </div>

                {/* Right Column: Generation */}
                <div className="space-y-6">
                  <section className="bg-white p-6 rounded-2xl border border-zinc-200 shadow-sm h-full flex flex-col">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <Terminal className="w-5 h-5 text-zinc-500" />
                        <h2 className="font-semibold text-zinc-900">2. Inference Output</h2>
                      </div>
                      <button
                        onClick={generateFromModel}
                        disabled={!modelRef.current || isGenerating || isTraining}
                        className="px-4 py-2 bg-emerald-600 text-white text-sm font-medium rounded-lg hover:bg-emerald-700 disabled:opacity-30 transition-all"
                      >
                        {isGenerating ? "Generating..." : "Generate Text"}
                      </button>
                    </div>
                    
                    <div className="flex-1 bg-zinc-50 rounded-xl p-4 font-mono text-sm text-zinc-800 whitespace-pre-wrap border border-zinc-100 min-h-[300px]">
                      {generatedText || (
                        <div className="h-full flex flex-col items-center justify-center text-zinc-400 text-center">
                          <Brain className="w-12 h-12 mb-2 opacity-20" />
                          <p>Train the model then click generate<br/>to see next-token prediction in action.</p>
                        </div>
                      )}
                    </div>
                  </section>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
      
      <footer className="px-6 py-3 bg-white border-t border-zinc-100 shrink-0">
        <p className="text-center text-[10px] text-zinc-400 uppercase tracking-widest font-medium">
          LLM Studio • Built with TensorFlow.js & Gemini 3.1
        </p>
      </footer>
    </div>
  );
}
