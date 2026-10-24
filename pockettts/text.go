package pockettts

import (
	"errors"
	"strings"
	"unicode"
	"unicode/utf8"
)

// The text handling of Kyutai's pocket_tts/models/text_chunking.py: the
// model is trained on single sentences, so text is split at sentence ends
// (and long sentences at commas) into chunks of at most 50 tokens, each
// spoken on its own from the voice's state.

var errEmpty = errors.New("pockettts: there is no text to speak")

// pySpace is Python's str.isspace, which also counts the four ASCII
// separators U+001C to U+001F.
func pySpace(r rune) bool { return unicode.IsSpace(r) || r >= 0x1c && r <= 0x1f }

func pyStrip(s string) string { return strings.TrimFunc(s, pySpace) }

func wordCount(s string) int { return len(strings.FieldsFunc(s, pySpace)) }

const (
	terminalPunct = ".!?…"
	weakPunct     = ",;:-–—"
	closers       = "\"'”’)]»"
)

// prepareText is prepare_text_prompt: tidy spaces, capitalize, end on a
// sentence mark. It returns the text and how many frames to make after
// the model says it's done, less the 2 generate_audio_stream adds.
func (c *config) prepareText(text string) (string, int, error) {
	text = pyStrip(text)
	if text == "" {
		return "", 0, errEmpty
	}
	text = strings.ReplaceAll(strings.ReplaceAll(text, "\n", " "), "\r", " ")
	text = strings.ReplaceAll(text, "  ", " ")
	if c.removeSemicolons {
		text = strings.ReplaceAll(text, ";", ",")
	}
	guess := 1
	if wordCount(text) <= 4 {
		guess = 3
	}
	if c.capitalize {
		r, n := utf8.DecodeRuneInString(text)
		if !unicode.IsUpper(r) {
			text = strings.ToUpper(text[:n]) + text[n:]
		}
	}
	if c.appendPunct {
		text = ensureTerminal(text)
	}
	if c.padShort && wordCount(text) < 5 {
		text = strings.Repeat(" ", 8) + text
	}
	return text, guess, nil
}

// ensureTerminal is _ensure_terminal_punctuation.
func ensureTerminal(text string) string {
	core := strings.TrimRight(text, closers+" ")
	cl := pyStrip(text[len(core):])
	if core == "" {
		return text
	}
	last, _ := utf8.DecodeLastRuneInString(core)
	switch {
	case strings.ContainsRune(terminalPunct, last):
		return text
	case strings.ContainsRune(weakPunct, last):
		return strings.TrimRight(core, weakPunct+" ") + "." + cl
	}
	return text + "."
}

// isDecimalPeriod is _is_decimal_period_boundary: does a segment
// starting at ids[at] begin right after the period of a decimal number?
func (t *tokenizer) isDecimalPeriod(ids []int, at int) bool {
	prefix, suffix := t.decode(ids[:at]), t.decode(ids[at:])
	last, n := utf8.DecodeLastRuneInString(prefix)
	before, _ := utf8.DecodeLastRuneInString(prefix[:len(prefix)-n])
	first, _ := utf8.DecodeRuneInString(suffix)
	return utf8.RuneCountInString(prefix) >= 2 && last == '.' && unicode.IsDigit(before) && suffix != "" && unicode.IsDigit(first)
}

// boundaries is _find_boundary_indices: segment starts after each run
// of boundary tokens, with 0 first and len(ids) last.
func (t *tokenizer) boundaries(ids []int, bound []int, skipDecimal bool) []int {
	in := func(id int) bool {
		for _, b := range bound {
			if b == id {
				return true
			}
		}
		return false
	}
	out := []int{0}
	prev := false
	for i, id := range ids {
		if in(id) {
			prev = true
			continue
		}
		if prev {
			if skipDecimal && t.isDecimalPeriod(ids, i) {
				prev = false
				continue
			}
			out = append(out, i)
		}
		prev = false
	}
	return append(out, len(ids))
}

type segment struct {
	n    int
	text string
}

func (t *tokenizer) segments(ids, bounds []int) []segment {
	var out []segment
	for i := 0; i+1 < len(bounds); i++ {
		a, b := bounds[i], bounds[i+1]
		out = append(out, segment{b - a, t.decode(ids[a:b])})
	}
	return out
}

// chunks is split_into_best_sentences.
func (m *Model) chunks(text string, maxTokens int) ([]string, error) {
	t := m.tok
	text, _, err := m.cfg.prepareText(text)
	if err != nil {
		return nil, err
	}
	ids := t.encode(pyStrip(text))
	sentences := t.segments(ids, t.boundaries(ids, t.encode(".!...?")[1:], true))
	weak := t.encode(",;:")[1:]
	var refined []segment
	for _, s := range sentences {
		if s.n <= maxTokens {
			refined = append(refined, s)
			continue
		}
		sub := t.encode(pyStrip(s.text))
		parts := t.segments(sub, t.boundaries(sub, weak, false))
		if len(parts) > 1 {
			refined = append(refined, parts...)
		} else {
			refined = append(refined, s)
		}
	}
	var out []string
	cur, n := "", 0
	for _, s := range refined {
		switch {
		case cur == "":
			cur, n = s.text, s.n
		case n+s.n > maxTokens:
			out = append(out, pyStrip(cur))
			cur, n = s.text, s.n
		default:
			cur += " " + s.text
			n += s.n
		}
	}
	if cur != "" {
		out = append(out, pyStrip(cur))
	}
	return out, nil
}
