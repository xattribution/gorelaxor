package conformer

import (
	"errors"
	"fmt"
	"regexp"

	"github.com/xattribution/gorelaxor/internal/gguf"
	"github.com/xattribution/gorelaxor/internal/kern"
)

// languagePrompt is Nemotron 3.5's language prompt (NeMo
// PromptStreamingMixin): each encoder frame, joined with a one-hot
// language vector, goes through Linear, ReLU, Linear, and the result
// replaces the frame before the decoder.
type languagePrompt struct {
	n      int // one-hot size (num_prompts)
	w0, w2 *kern.Weight
	b0, b2 []float32
	index  map[string]int // locale (and NeMo's aliases) to one-hot slot
	order  []string       // the locales in the file's order
	auto   int            // the slot for "auto", or -1
	tags   map[int]string // language-tag token to its locale
}

// langTag matches the tokens a prompt model writes after a sentence to
// say which language it heard, such as <de-DE>.
var langTag = regexp.MustCompile(`^<([a-z]{2,3}-[A-Za-z]{2,4})>$`)

// loadPrompt reads the prompt network and its language table, if the
// file has one (stt.parakeet.prompt.*, transcribe.cpp's names).
func (m *Model) loadPrompt(f *gguf.File, l *loader) error {
	p := "stt.parakeet.prompt."
	n := f.Int(p+"num_prompts", 0)
	if n <= 0 {
		if f.Has("prompt.mlp.0.weight") {
			return errors.New("conformer: the model has a prompt network but no prompt settings")
		}
		return nil
	}
	if a := f.String(p+"activation", "relu"); a != "relu" {
		return fmt.Errorf("conformer: prompt activation %q is not supported (only relu)", a)
	}
	lp := &languagePrompt{
		n:  n,
		w0: l.mat("prompt.mlp.0.weight", false), b0: l.vec("prompt.mlp.0.bias", false),
		w2: l.mat("prompt.mlp.2.weight", false), b2: l.vec("prompt.mlp.2.bias", false),
		index: map[string]int{}, auto: f.Int(p+"auto_id", -1), tags: map[int]string{},
	}
	if l.err != nil {
		return l.err
	}
	D, H := m.HP.DModel, f.Int(p+"hidden", 0)
	if lp.w0.Cols != D+n || lp.w0.Rows != H || len(lp.b0) != H || lp.w2.Cols != H || lp.w2.Rows != D || len(lp.b2) != D {
		return errors.New("conformer: the prompt network doesn't fit its settings")
	}
	locales, indices := f.Strings(p+"dictionary.locales"), f.Ints(p+"dictionary.indices")
	if len(locales) == 0 || len(locales) != len(indices) {
		return errors.New("conformer: the model's language table is missing or broken")
	}
	for i, loc := range locales {
		if loc == "" || indices[i] < 0 || indices[i] >= n {
			return fmt.Errorf("conformer: language table entry %q is out of range", loc)
		}
		if _, dup := lp.index[loc]; !dup {
			lp.order = append(lp.order, loc)
		}
		lp.index[loc] = indices[i]
	}
	if lp.auto >= n {
		return fmt.Errorf("conformer: the auto language slot %d is out of range", lp.auto)
	}
	for id, piece := range m.Vocab {
		if s := langTag.FindStringSubmatch(piece); s != nil {
			lp.tags[id] = s[1]
		}
	}
	m.prompt = lp
	m.lang = lp.auto
	if m.lang < 0 {
		m.lang = indices[0] // transcribe.cpp's choice without an auto slot
	}
	return nil
}

// Languages lists the locales a multilingual model takes, such as
// "en-US", "de-DE" and "auto", with NeMo's short aliases ("de"). It is
// nil for single-language models.
func (m *Model) Languages() []string {
	if m.prompt == nil {
		return nil
	}
	return append([]string(nil), m.prompt.order...)
}

// WithLanguage returns the model set to transcribe one language: a
// locale from Languages, or "auto" (the default) to let the model tell.
// It shares the loaded weights, so it costs nothing to keep one per
// language. Single-language models take only "" and "auto".
func (m *Model) WithLanguage(locale string) (*Model, error) {
	if locale == "" {
		locale = "auto"
	}
	if m.prompt == nil {
		if locale == "auto" {
			return m, nil
		}
		return nil, fmt.Errorf("conformer: this model transcribes one language and takes no %q", locale)
	}
	i, ok := m.prompt.index[locale]
	if !ok && locale == "auto" && m.prompt.auto >= 0 {
		i, ok = m.prompt.auto, true
	}
	if !ok {
		return nil, fmt.Errorf("conformer: the model has no language %q", locale)
	}
	c := *m
	c.lang = i
	return &c, nil
}

// DetectedLanguage is the locale of the last language tag in toks
// ("de-DE"), or "" if there is none. Nemotron 3.5 writes a tag after
// each sentence's final punctuation; Text and Words leave tags out.
func (m *Model) DetectedLanguage(toks []Token) string {
	if m.prompt == nil {
		return ""
	}
	for i := len(toks) - 1; i >= 0; i-- {
		if loc, ok := m.prompt.tags[toks[i].ID]; ok {
			return loc
		}
	}
	return ""
}

// IsLanguageTag reports whether a token is one of the language tags a
// multilingual model writes after each sentence (DetectedLanguage), for
// callers that join tokens into words themselves.
func (m *Model) IsLanguageTag(id int) bool { return m.isLangTag(id) }

// isLangTag reports whether id is a language tag of a prompt model.
func (m *Model) isLangTag(id int) bool {
	if m.prompt == nil {
		return false
	}
	_, ok := m.prompt.tags[id]
	return ok
}

// applyPrompt runs encoder output x [L][d_model] through the language
// prompt, as NeMo's forward and _apply_prompt_to_encoded do, and returns
// what the decoder reads. Models without a prompt return x.
func (m *Model) applyPrompt(x []float32, L, threads int) []float32 {
	lp := m.prompt
	if lp == nil || L == 0 {
		return x
	}
	D := m.HP.DModel
	in := make([]float32, L*(D+lp.n))
	for t := 0; t < L; t++ {
		row := in[t*(D+lp.n):]
		copy(row, x[t*D:(t+1)*D])
		row[D+m.lang] = 1
	}
	h := make([]float32, L*lp.w0.Rows)
	kern.Linear(h, in, L, lp.w0, lp.b0, threads)
	relu(h)
	y := make([]float32, L*D)
	kern.Linear(y, h, L, lp.w2, lp.b2, threads)
	return y
}
