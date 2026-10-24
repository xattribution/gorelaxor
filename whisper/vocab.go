package whisper

import (
	"regexp"
	"strings"
)

// Language codes in the order of the model's language tokens.
var langCodes = strings.Fields(`en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue`)

type vocab struct {
	words        []string
	toID         map[string]int
	n            int
	multilingual bool
	nLang        int

	eot, sot, translate, transcribe, solm, prev, nosp, not, beg int
}

func newVocab(words []string, n int) *vocab {
	v := &vocab{words: words, n: n, toID: make(map[string]int, len(words))}
	for i, w := range words {
		if _, dup := v.toID[w]; !dup {
			v.toID[w] = i
		}
	}
	v.eot, v.sot = 50256, 50257
	v.translate, v.transcribe, v.solm, v.prev, v.nosp, v.not, v.beg = 50357, 50358, 50359, 50360, 50361, 50362, 50363
	v.multilingual = n >= 51865
	if v.multilingual {
		v.nLang = n - 51765 - 1
		dt := v.nLang - 98
		v.eot++
		v.sot++
		v.translate += dt
		v.transcribe += dt
		v.solm += dt
		v.prev += dt
		v.nosp += dt
		v.not += dt
		v.beg += dt
	}
	return v
}

// langToken is the token for a language code, or -1.
func (v *vocab) langToken(code string) int {
	if !v.multilingual {
		return -1
	}
	for i, c := range langCodes {
		if c == code && i < v.nLang {
			return v.sot + 1 + i
		}
	}
	return -1
}

func (v *vocab) isLang(t int) bool { return v.multilingual && t > v.sot && t <= v.sot+v.nLang }

// text is the bytes a text token stands for; special tokens give "".
func (v *vocab) text(t int) string {
	if t < 0 || t >= v.eot || t >= len(v.words) {
		return ""
	}
	return v.words[t]
}

var reWords = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+`)

// tokenize turns text into tokens, the longest known piece first, the way
// whisper.cpp does for prompts.
func (v *vocab) tokenize(s string) []int {
	var out []int
	for _, w := range reWords.FindAllString(s, -1) {
		for i := 0; i < len(w); {
			j := len(w)
			for ; j > i; j-- {
				if id, ok := v.toID[w[i:j]]; ok && id < v.eot {
					out = append(out, id)
					break
				}
			}
			if j == i {
				i++ // a byte no token covers
			} else {
				i = j
			}
		}
	}
	return out
}

// Tokens whisper.cpp keeps out of transcripts unless asked: brackets and
// symbols that mark sounds rather than speech.
var nonSpeech = []string{"\"", "#", "(", ")", "*", "+", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^",
	"_", "`", "{", "|", "}", "~", "「", "」", "『", "』", "<<", ">>", "<<<", ">>>", "--",
	"---", "-(", "-[", "('", "(\"", "((", "))", "(((", ")))", "[[", "]]", "{{", "}}", "♪♪",
	"♪♪♪", "♩", "♪", "♫", "♬", "♭", "♮", "♯"}

// suppressed lists the tokens never sampled in a transcript.
func (v *vocab) suppressed() []int {
	var out []int
	for _, t := range nonSpeech {
		for _, s := range []string{t, " " + t} {
			if id, ok := v.toID[s]; ok && id < v.eot {
				out = append(out, id)
			}
		}
	}
	for _, s := range []string{" -", " '"} {
		if id, ok := v.toID[s]; ok {
			out = append(out, id)
		}
	}
	return out
}
