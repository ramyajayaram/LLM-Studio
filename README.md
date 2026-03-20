# LLM Studio

LLM Studio is a comprehensive development environment for exploring and building Large Language Models. It combines the power of Google's state-of-the-art Gemini models with a hands-on playground for building custom Transformer architectures from scratch.

## Features

### 1. Gemini Chat Pro
A high-performance, modern chat interface powered by **Gemini 3.1 Flash**.
- **Real-time Streaming:** Instant responses as they are generated.
- **Markdown Support:** Rich text rendering for code, lists, and more.
- **Polished UI:** A clean, professional design with smooth animations.

### 2. Model Playground (Toy GPT)
A character-level Transformer model implemented in **TensorFlow.js** that trains directly in your browser.
- **Custom Architecture:** Features Token Embedding, Positional Encoding, Self-Attention, and Feed-Forward layers.
- **Live Training:** Train the model on any text input and watch the loss decrease in real-time.
- **Inference:** Generate new text based on the patterns learned during training.

## Tech Stack

- **Frontend:** React 19, Tailwind CSS 4, Motion
- **AI Models:** Gemini 3.1 Flash (via `@google/genai`)
- **Machine Learning:** TensorFlow.js (`@tensorflow/tfjs`)
- **Icons:** Lucide React

## Getting Started

1. **Chat:** Use the "Chat" tab to interact with Gemini. Ensure your `GEMINI_API_KEY` is configured in the AI Studio secrets.
2. **Playground:** Switch to the "Model Playground" tab.
3. **Train:** Paste some text (e.g., Shakespeare) into the training data area and click "Train Toy GPT Model".
4. **Generate:** Once training is complete, click "Generate Text" to see the model's predictions.


