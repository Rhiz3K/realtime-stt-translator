// SentencePiece detokeniser for the Parakeet-tdt-0.6b-v3 engine.
//
// vocab.txt (from the ONNX export) is one entry per line: "<piece> <id>", e.g.
//   ▁the 42
//   <blk> 8192
// Detok mirrors the Nemotron tokeniser: concatenate pieces, turn the
// SentencePiece word-boundary marker ▁ (U+2581) into a space, skip control /
// special tokens (<unk>, <blk>, <|nospeech|>, language/PnC tags…), then trim.

const WORD_MARK = "▁"; // ▁

export class Tokenizer {
  constructor(pieces) {
    this.pieces = pieces; // id -> piece
  }

  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`vocab fetch failed: ${res.status}. Parakeet model assets are not ready on the server yet.`);
    }
    const text = await res.text();
    const pieces = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      // id is the last whitespace-separated field; the piece never contains a
      // raw space (SentencePiece uses ▁ and <0x..> byte fallbacks).
      const sp = line.lastIndexOf(" ");
      if (sp < 0) continue;
      const piece = line.slice(0, sp);
      const id = Number.parseInt(line.slice(sp + 1), 10);
      if (Number.isInteger(id) && id >= 0) pieces[id] = piece;
    }
    if (!pieces.length) throw new Error("vocab.txt parsed empty");
    return new Tokenizer(pieces);
  }

  /** Map a single token id to its raw piece (empty string if out of range). */
  piece(id) {
    return id >= 0 && id < this.pieces.length && this.pieces[id] != null ? this.pieces[id] : "";
  }

  /** Detokenise a list of token ids into display text. */
  decode(ids) {
    let s = "";
    for (const id of ids) {
      const p = this.piece(id);
      // Skip control / special tokens: <unk>, <blk>, <|nospeech|>, <|pnc|>,
      // <|startoftranscript|>, language tags, etc.
      if (/^<[^>]*>$/.test(p)) continue;
      s += p;
    }
    return s.split(WORD_MARK).join(" ").replace(/\s+/g, " ").trim();
  }
}
