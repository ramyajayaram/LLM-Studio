/**
 * A simple character-level tokenizer for our toy LLM.
 */
export class Tokenizer {
  private charToId: Map<string, number> = new Map();
  private idToChar: Map<number, string> = new Map();
  public vocabSize: number = 0;

  constructor(text: string) {
    const chars = Array.from(new Set(text)).sort();
    this.vocabSize = chars.length;
    chars.forEach((char, i) => {
      this.charToId.set(char, i);
      this.idToChar.set(i, char);
    });
  }

  encode(text: string): number[] {
    return Array.from(text).map(c => this.charToId.get(c) ?? 0);
  }

  decode(ids: number[]): string {
    return ids.map(id => this.idToChar.get(id) ?? "").join("");
  }
}
