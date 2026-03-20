import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

const apiKey = process.env.GEMINI_API_KEY;

if (!apiKey) {
  console.warn("GEMINI_API_KEY is not set. Chat features will not work.");
}

const ai = new GoogleGenAI({ apiKey: apiKey || "" });

export interface Message {
  role: "user" | "model";
  content: string;
}

export async function* sendMessageStream(messages: Message[]) {
  if (!apiKey) {
    yield "Error: Gemini API key is missing. Please configure it in the settings.";
    return;
  }

  try {
    const chat = ai.chats.create({
      model: "gemini-3.1-flash-preview",
      config: {
        systemInstruction: "You are a helpful, creative, and clever AI assistant. Provide clear, concise, and accurate responses. Use markdown for formatting when appropriate.",
      },
    });

    // Send the history first, then the last message
    // Actually, sendMessageStream takes a message string.
    // We can pass history to create() if we want, but for simplicity:
    const lastMessage = messages[messages.length - 1].content;
    
    const result = await chat.sendMessageStream({
      message: lastMessage,
    });

    for await (const chunk of result) {
      const text = chunk.text;
      if (text) {
        yield text;
      }
    }
  } catch (error) {
    console.error("Gemini API Error:", error);
    yield "I encountered an error while processing your request. Please try again.";
  }
}
