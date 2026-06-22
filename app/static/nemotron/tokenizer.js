// SentencePiece detokeniser for the Nemotron-3.5-ASR engine.
//
// vocab.json is a flat array (id -> piece) extracted from tokenizer.model by
// scripts/prepare_nemotron_onnx.py. Detok mirrors the Python reference
// (scripts/nemotron_reference.py main()): concatenate pieces, turn the
// SentencePiece word-boundary marker ▁ (U+2581) into a space, then trim.

const WORD_MARK = "▁"; // ▁

export class Tokenizer {
  constructor(pieces) {
    this.pieces = pieces;
  }

  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`vocab fetch failed: ${res.status}`);
    return new Tokenizer(await res.json());
  }

  /** Map a single token id to its raw piece (empty string if out of range). */
  piece(id) {
    return id >= 0 && id < this.pieces.length ? this.pieces[id] : "";
  }

  /** Detokenise a list of token ids into display text. */
  decode(ids) {
    let s = "";
    for (const id of ids) {
      const p = this.piece(id);
      // Skip control / special tokens: <unk>, language tags like <en-US> that the
      // multilingual model emits at segment ends, etc.
      if (/^<[^>]*>$/.test(p)) continue;
      s += p;
    }
    return s.split(WORD_MARK).join(" ").replace(/\s+/g, " ").trim();
  }
}
