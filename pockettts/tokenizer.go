package pockettts

import (
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/xattribution/gorelaxor/internal/gguf"
)

// tokenizer is Pocket TTS's tokenizer.json as Hugging Face tokenizers
// runs it: special tokens matched in the raw text, a ▁ put in front of
// each other stretch (the Prepend normalizer), spaces turned into ▁ and
// the text split before each run of them (Metaspace, prepend "always",
// split), then a SentencePiece unigram model's best path through each
// piece, with characters it has no piece for spelled as <0xNN> bytes.
// tools/convert_pocket_tts.py refuses a tokenizer.json that does
// anything else.
type tokenizer struct {
	pieces   []string
	scores   []float64
	ids      map[string]int
	maxLen   int // the longest piece, in bytes
	minScore float64
	unk      int
	bytes    [256]int // <0xNN>'s id, -1 if missing
	special  []int    // ids of the special tokens, longest first
	isSpec   []bool
}

const (
	meta     = "▁"
	unkExtra = 10 // tokenizers' K_UNK_PENALTY: an unknown character costs the lowest score less 10
)

func loadTokenizer(f *gguf.File) (*tokenizer, error) {
	if f.String("tokenizer.ggml.model", "") != "unigram" || !f.Bool("tokenizer.ggml.byte_fallback", false) {
		return nil, errors.New("pockettts: the file's tokenizer isn't a byte-fallback unigram model")
	}
	t := &tokenizer{pieces: f.Strings("tokenizer.ggml.tokens"), ids: map[string]int{}}
	raw, _ := f.KV["tokenizer.ggml.scores"].([]any)
	if len(t.pieces) == 0 || len(raw) != len(t.pieces) {
		return nil, errors.New("pockettts: the tokenizer's pieces and scores don't match")
	}
	t.scores = make([]float64, len(raw))
	for i, v := range raw {
		s, ok := v.(float64)
		if !ok {
			return nil, errors.New("pockettts: the tokenizer's scores aren't float64")
		}
		t.scores[i] = s
		if i == 0 || s < t.minScore {
			t.minScore = s
		}
	}
	for i := range t.bytes {
		t.bytes[i] = -1
	}
	for i, p := range t.pieces {
		if _, dup := t.ids[p]; !dup {
			t.ids[p] = i
		}
		t.maxLen = max(t.maxLen, len(p))
		var b byte
		if len(p) == 6 && strings.HasPrefix(p, "<0x") && p[5] == '>' {
			if _, err := fmt.Sscanf(p[3:5], "%02X", &b); err == nil && fmt.Sprintf("<0x%02X>", b) == p {
				t.bytes[b] = i
			}
		}
	}
	t.unk = f.Int("tokenizer.ggml.unknown_token_id", 0)
	if t.unk < 0 || t.unk >= len(t.pieces) {
		return nil, errors.New("pockettts: the tokenizer's unknown token is out of range")
	}
	t.isSpec = make([]bool, len(t.pieces))
	for _, id := range f.Ints("tokenizer.ggml.special_ids") {
		if id < 0 || id >= len(t.pieces) || t.pieces[id] == "" {
			return nil, errors.New("pockettts: a special token is out of range")
		}
		t.special = append(t.special, id)
		t.isSpec[id] = true
	}
	return t, nil
}

// encode returns text's token ids.
func (t *tokenizer) encode(text string) []int {
	var out []int
	start := 0
	for i := 0; i < len(text); {
		id, n := t.specialAt(text[i:])
		if n == 0 {
			i++
			continue
		}
		out = t.encodePlain(out, text[start:i])
		out = append(out, id)
		i += n
		start = i
	}
	return t.encodePlain(out, text[start:])
}

// specialAt is the longest special token text starts with (the
// leftmost-longest match tokenizers' added vocabulary makes).
func (t *tokenizer) specialAt(s string) (id, n int) {
	for _, sid := range t.special {
		if p := t.pieces[sid]; len(p) > n && strings.HasPrefix(s, p) {
			id, n = sid, len(p)
		}
	}
	return id, n
}

func (t *tokenizer) encodePlain(out []int, s string) []int {
	if s == "" {
		return out
	}
	s = strings.ReplaceAll(meta+s, " ", meta)
	// Split before each run of ▁ (MergedWithNext: a run of delimiters
	// starts one piece together with what follows it).
	start := 0
	prevMeta := false
	for i := 0; i < len(s); {
		isMeta := strings.HasPrefix(s[i:], meta)
		if isMeta && !prevMeta && i > start {
			out = t.viterbi(out, s[start:i])
			start = i
		}
		prevMeta = isMeta
		if isMeta {
			i += len(meta)
		} else {
			_, n := utf8.DecodeRuneInString(s[i:])
			i += n
		}
	}
	return t.viterbi(out, s[start:])
}

// viterbi appends the unigram model's best tokenization of one piece:
// tokenizers' Unigram.encode_optimized, unknown characters fused, then
// spelled as bytes.
func (t *tokenizer) viterbi(out []int, s string) []int {
	type node struct {
		id    int
		score float64
		from  int // -1: not reached
	}
	n := len(s)
	best := make([]node, n+1)
	for i := range best {
		best[i].from = -1
	}
	unkScore := t.minScore - unkExtra
	for at := 0; at < n; {
		base := best[at].score
		_, mb := utf8.DecodeRuneInString(s[at:])
		single := false
		for l := 1; l <= t.maxLen && at+l <= n; l++ {
			id, ok := t.ids[s[at:at+l]]
			if !ok {
				continue
			}
			c := base + t.scores[id]
			if nd := &best[at+l]; nd.from < 0 || c > nd.score {
				*nd = node{id, c, at}
			}
			if l == mb {
				single = true
			}
		}
		if !single {
			c := base + unkScore
			if nd := &best[at+mb]; nd.from < 0 || c > nd.score {
				*nd = node{t.unk, c, at}
			}
		}
		at += mb
	}
	// Walk back, joining runs of unknown characters into one string.
	var parts []string
	unkRun := ""
	for end := n; end > 0; {
		nd := best[end]
		if nd.id == t.unk {
			unkRun = s[nd.from:end] + unkRun
		} else {
			if unkRun != "" {
				parts = append(parts, unkRun)
				unkRun = ""
			}
			parts = append(parts, s[nd.from:end])
		}
		end = nd.from
	}
	if unkRun != "" {
		parts = append(parts, unkRun)
	}
	for i := len(parts) - 1; i >= 0; i-- {
		p := parts[i]
		if id, ok := t.ids[p]; ok {
			out = append(out, id)
			continue
		}
		ids := make([]int, 0, len(p))
		for j := 0; j < len(p) && ids != nil; j++ {
			if b := t.bytes[p[j]]; b >= 0 {
				ids = append(ids, b)
			} else {
				ids = nil
			}
		}
		if ids == nil {
			ids = []int{t.unk}
		}
		out = append(out, ids...)
	}
	return out
}

// decode turns ids back into text, special tokens left out: tokenizers'
// decode with its Metaspace, ByteFallback and Fuse decoders.
func (t *tokenizer) decode(ids []int) string {
	var sb strings.Builder
	var bytesRun []byte
	flush := func() {
		if len(bytesRun) == 0 {
			return
		}
		if utf8.Valid(bytesRun) {
			sb.Write(bytesRun)
		} else {
			sb.WriteString(strings.Repeat("�", len(bytesRun)))
		}
		bytesRun = bytesRun[:0]
	}
	first := true
	for _, id := range ids {
		if id < 0 || id >= len(t.pieces) || t.isSpec[id] {
			continue
		}
		p := t.pieces[id]
		if first {
			p = strings.ReplaceAll(p, meta, "") // the first token loses every ▁
		} else {
			p = strings.ReplaceAll(p, meta, " ")
		}
		first = false
		if len(p) == 6 && strings.HasPrefix(p, "<0x") && p[5] == '>' {
			var b byte
			if _, err := fmt.Sscanf(p[3:5], "%02X", &b); err == nil {
				bytesRun = append(bytesRun, b)
				continue
			}
		}
		flush()
		sb.WriteString(p)
	}
	flush()
	return sb.String()
}
