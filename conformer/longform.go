package conformer

import (
	"errors"
	"math"
	"strings"
)

// Word is one transcribed word with its time in the audio.
type Word struct {
	Text       string
	Start, End float64 // seconds
	P          float32 // lowest token probability in the word
}

// FrameSeconds is the time one encoder frame covers (80 ms).
func (m *Model) FrameSeconds() float64 {
	return float64(m.Mel.cfg.Hop*m.HP.Subsampling) / float64(m.Mel.cfg.SampleRate)
}

// Words groups tokens into words: a SentencePiece piece starting with
// "▁" begins a new one.
func (m *Model) Words(toks []Token) []Word {
	fs := m.FrameSeconds()
	var out []Word
	for _, t := range toks {
		if t.ID < 0 || t.ID >= len(m.Vocab) || t.ID == m.HP.Blank || m.isLangTag(t.ID) {
			continue
		}
		piece := m.Vocab[t.ID]
		start := float64(t.Frame) * fs
		end := float64(t.Frame+max(t.Duration, 1)) * fs
		if strings.HasPrefix(piece, "▁") || len(out) == 0 {
			out = append(out, Word{Text: strings.TrimPrefix(piece, "▁"), Start: start, End: end, P: t.P})
			continue
		}
		w := &out[len(out)-1]
		w.Text += piece
		w.End = max(w.End, end)
		w.P = min(w.P, t.P)
	}
	kept := out[:0]
	for _, w := range out {
		if w.Text != "" {
			kept = append(kept, w)
		}
	}
	return kept
}

// SegmentParams set how a long recording is cut at pauses for the final
// pass.
type SegmentParams struct {
	MaxSeconds float64 // no piece longer than this
	MinSeconds float64 // no cut before this far into a piece
	MinPause   float64 // the shortest silence worth cutting in
}

// DefaultSegments keep pieces short enough for full attention to stay
// cheap on a CPU while giving the model whole sentences.
var DefaultSegments = SegmentParams{MaxSeconds: 30, MinSeconds: 5, MinPause: 0.2}

// valid falls back to DefaultSegments for settings that can't work (zero,
// negative, NaN, or pieces under a second).
func (p SegmentParams) valid() SegmentParams {
	if !(p.MaxSeconds >= 1) || math.IsInf(p.MaxSeconds, 0) || !(p.MinSeconds >= 0) || p.MinSeconds >= p.MaxSeconds || !(p.MinPause >= 0) {
		return DefaultSegments
	}
	return p
}

// Segment is a piece of the recording, in samples.
type Segment struct{ Start, End int }

// Segments cuts n samples into pieces at pauses between speech regions
// (from the VAD), as transcribe.cpp's long-form segmenter does: each
// cut goes in the middle of the last pause that ends before MaxSeconds
// and starts after MinSeconds, else at the last pause whose middle falls
// there, else at MaxSeconds. Pieces with no speech are left out.
func Segments(n, rate int, speech []Region, p SegmentParams) []Segment {
	if n <= 0 || rate <= 0 {
		return nil
	}
	p = p.valid()
	maxN := int(p.MaxSeconds * float64(rate))
	if n <= maxN {
		return []Segment{{0, n}}
	}
	sec := func(s int) float64 { return float64(s) / float64(rate) }
	hasSpeech := func(a, b float64) bool {
		for _, r := range speech {
			if r.End > a && r.Start < b {
				return true
			}
		}
		return false
	}
	var out []Segment
	pos := 0
	for pos < n {
		if n-pos <= maxN {
			if hasSpeech(sec(pos), sec(n)) {
				out = append(out, Segment{pos, n})
			}
			break
		}
		base := sec(pos)
		lo, hi := base+p.MinSeconds, base+p.MaxSeconds
		// pauses: gaps between speech regions after pos
		var pauses []Region
		prev := base
		for _, r := range speech {
			if r.End <= base {
				continue
			}
			if r.Start-prev >= p.MinPause-1e-6 {
				pauses = append(pauses, Region{prev, r.Start})
			}
			prev = max(prev, r.End)
		}
		if sec(n)-prev >= p.MinPause-1e-6 {
			pauses = append(pauses, Region{prev, sec(n)})
		}
		cut := hi
		found := false
		for _, r := range pauses {
			if r.Start >= lo && r.End <= hi {
				cut, found = (r.Start+r.End)/2, true
			}
		}
		if !found {
			for _, r := range pauses {
				if mid := (r.Start + r.End) / 2; mid >= lo && mid <= hi {
					cut = mid
				}
			}
		}
		end := min(n, int(cut*float64(rate)+0.5))
		if end <= pos {
			end = min(n, pos+maxN) // maxN is at least a second, so this always moves on
		}
		if hasSpeech(base, sec(end)) {
			out = append(out, Segment{pos, end})
		}
		pos = end
	}
	return out
}

// Transcribe runs the final pass over a whole recording (16 kHz mono):
// cut at pauses when longer than one piece, each piece encoded and
// decoded on its own, the tokens' frames counted from the recording's
// start. vad may be nil for recordings of one piece.
func (m *Model) Transcribe(pcm []float32, vad *VAD, p SegmentParams, threads int) ([]Token, error) {
	rate := m.Mel.cfg.SampleRate
	pcm, p = finite(pcm), p.valid()
	segs := []Segment{{0, len(pcm)}}
	if float64(len(pcm)) > p.MaxSeconds*float64(rate) {
		if vad == nil {
			return nil, errors.New("conformer: a recording this long needs the silence detector to cut it")
		}
		probs, err := vad.ProbsLong(pcm, threads)
		if err != nil {
			return nil, err
		}
		segs = Segments(len(pcm), rate, vad.Speech(probs, DefaultSpeech), p)
	}
	hopFrames := m.Mel.cfg.Hop * m.HP.Subsampling
	var out []Token
	for _, sg := range segs {
		// start on a whole encoder frame so token times stay exact; the
		// cut is in a pause, so the extra audio is silence
		start := sg.Start / hopFrames * hopFrames
		piece := pcm[start:sg.End]
		if len(piece) < 2*m.Mel.cfg.Hop {
			continue
		}
		mel, frames, err := m.Mel.Compute(piece, threads)
		if err != nil {
			return nil, err
		}
		enc, T := m.Encode(mel, frames, m.Mel.Valid(len(piece)), threads)
		out = append(out, m.Decode(enc, T, nil, start/hopFrames, threads)...)
	}
	return out, nil
}
