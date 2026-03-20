import * as tf from '@tensorflow/tfjs';

/**
 * A minimal Transformer-based LLM implemented in TensorFlow.js
 * This is a "Toy GPT" architecture.
 */
export class ToyGPT {
  private model: tf.LayersModel;
  private vocabSize: number;
  private maxLen: number;
  private embedDim: number;

  constructor(vocabSize: number, maxLen: number = 32, embedDim: number = 64) {
    this.vocabSize = vocabSize;
    this.maxLen = maxLen;
    this.embedDim = embedDim;

    const input = tf.input({ shape: [maxLen] });
    
    // Token Embedding
    let x = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: embedDim,
      maskZero: false
    }).apply(input) as tf.SymbolicTensor;

    // Positional Embedding (Simple additive)
    const posIndices = tf.range(0, maxLen).expandDims(0);
    const posEmbed = tf.layers.embedding({
      inputDim: maxLen,
      outputDim: embedDim
    }).apply(posIndices) as tf.SymbolicTensor;
    
    x = tf.layers.add().apply([x, posEmbed]) as tf.SymbolicTensor;

    // Simplified Attention (Single Head for robustness)
    const query = tf.layers.dense({ units: embedDim }).apply(x) as tf.SymbolicTensor;
    const key = tf.layers.dense({ units: embedDim }).apply(x) as tf.SymbolicTensor;
    const value = tf.layers.dense({ units: embedDim }).apply(x) as tf.SymbolicTensor;
    
    // Attention = softmax(Q @ K.T / sqrt(dk)) @ V
    // In TF.js layers, we can use a custom layer or just a dense block for this toy version
    // To keep it simple and lint-safe, we'll use a dense-based self-attention approximation
    const attn = tf.layers.dense({ units: embedDim, activation: 'tanh' }).apply(x) as tf.SymbolicTensor;
    const attnWeights = tf.layers.dense({ units: embedDim, activation: 'softmax' }).apply(attn) as tf.SymbolicTensor;
    const context = tf.layers.multiply().apply([attnWeights, value]) as tf.SymbolicTensor;
    
    x = tf.layers.layerNormalization().apply(tf.layers.add().apply([x, context])) as tf.SymbolicTensor;

    // Feed Forward
    const ffOutput = tf.layers.dense({ units: embedDim * 2, activation: 'relu' }).apply(x) as tf.SymbolicTensor;
    const ffOutputFinal = tf.layers.dense({ units: embedDim }).apply(ffOutput) as tf.SymbolicTensor;
    
    x = tf.layers.layerNormalization().apply(tf.layers.add().apply([x, ffOutputFinal])) as tf.SymbolicTensor;

    // Output Head
    const logits = tf.layers.dense({ units: vocabSize }).apply(x) as tf.SymbolicTensor;
    
    this.model = tf.model({ inputs: input, outputs: logits });
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'sparseCategoricalCrossentropy'
    });
  }

  async train(data: number[], epochs: number = 10, onEpochEnd?: (epoch: number, loss: number) => void) {
    const xData: number[][] = [];
    const yData: number[][] = [];

    for (let i = 0; i < data.length - this.maxLen; i++) {
      xData.push(data.slice(i, i + this.maxLen));
      yData.push(data.slice(i + 1, i + this.maxLen + 1));
    }

    const xs = tf.tensor2d(xData);
    const ys = tf.tensor3d(yData.map(y => y.map(val => [val])), [yData.length, this.maxLen, 1]);

    await this.model.fit(xs, ys, {
      epochs,
      batchSize: 32,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (onEpochEnd && logs) onEpochEnd(epoch, logs.loss);
        }
      }
    });

    xs.dispose();
    ys.dispose();
  }

  async generate(seed: number[], length: number = 50): Promise<number[]> {
    const generated = [...seed];
    for (let i = 0; i < length; i++) {
      const input = generated.slice(-this.maxLen);
      // Pad if shorter
      const paddedInput = [...input];
      while (paddedInput.length < this.maxLen) paddedInput.unshift(0);
      
      const inputTensor = tf.tensor2d([paddedInput]);
      const prediction = this.model.predict(inputTensor) as tf.Tensor;
      
      // Get the last token's logits
      const lastTokenLogits = prediction.slice([0, this.maxLen - 1, 0], [1, 1, this.vocabSize]);
      const reshapedLogits = lastTokenLogits.reshape([this.vocabSize]) as tf.Tensor1D;
      const probs = tf.softmax(reshapedLogits);
      const nextToken = tf.multinomial(probs, 1).dataSync()[0];
      
      generated.push(nextToken);
      
      inputTensor.dispose();
      prediction.dispose();
      probs.dispose();
    }
    return generated;
  }
}
