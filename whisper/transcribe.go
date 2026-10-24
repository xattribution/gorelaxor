package whisper

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"runtime"
	"strings"
	"time"
)

// Options control a transcription.
type Options struct {
	Language   string // "" is English for .en models and detected for the others; or "en", "es", ...
	Threads    int    // 0 uses every core but one
	Prompt     string // words to expect: names, acronyms, domain terms
	Translate  bool   // multilingual models: English text of speech in another language
	OnSegment  func(Segment)
	OnProgress func(done, total time.Duration)
}

// Segment is a stretch of speech and its text.
type Segment struct {
	Start, End time.Duration
	Text       string
}

func (o Options) threads() int {
	if o.Threads > 0 {
		return o.Threads
	}
	return max(1, runtime.NumCPU()-1)
}

// Decoding settings, whisper.cpp's defaults.
const (
	deltaMin       = 10 // frames (100 ms): less than this left is the end
	entropyThold   = 2.4
	logprobThold   = -1.0
	noSpeechThold  = 0.6
	maxInitialTS   = 1.0 // seconds
	bestOf         = 2   // decoders when sampling above temperature 0
	condTempCutoff = 0.5 // above this, earlier text isn't used as context
)

var temperatures = []float32{0, 0.2, 0.4, 0.6, 0.8, 1.0}

var ErrClosed = errors.New("whisper: the model was closed")

type tok struct {
	id, tid int
	p, plog float32
}

type seq struct {
	tokens                []tok
	cache                 *kvCache
	logits                []float32 // processed for the next sample
	probs, logprobs       []float32
	seekDelta, resultLen  int
	hasTS, failed, done   bool
	avgLogprob, score, en float64
}

// session carries what one transcription keeps from window to window.
type session struct {
	m        *Model
	opt      Options
	threads  int
	ws       *workspace
	past     []int // earlier text, the next window's context
	init     []int // start-of-transcript, language and task
	langSet  bool
	suppress []int
	rng      *rand.Rand
	seqs     []*seq
}

func (m *Model) newSession(opt Options) (*session, error) {
	s := &session{m: m, opt: opt, threads: opt.threads(), ws: &workspace{}, suppress: m.voc.suppressed(), rng: rand.New(rand.NewPCG(1, 2))}
	v := m.voc
	s.init = []int{v.sot}
	if v.multilingual {
		lang := strings.ToLower(strings.TrimSpace(opt.Language))
		if lang != "" && lang != "auto" {
			t := v.langToken(lang)
			if t < 0 {
				return nil, errors.New("whisper: unknown language " + opt.Language)
			}
			s.init = append(s.init, t, s.task())
			s.langSet = true
		}
	} else {
		s.langSet = true
	}
	if p := strings.TrimSpace(opt.Prompt); p != "" {
		toks := v.tokenize(" " + p)
		if len(toks) > m.hp.nTextCtx/2-1 {
			toks = toks[len(toks)-(m.hp.nTextCtx/2-1):]
		}
		s.past = toks
	}
	return s, nil
}

// frames is the number of 10 ms frames in n samples, as whisper.cpp counts.
func frames(n int) int {
	if n < nFFTWin/2 {
		return 0
	}
	return 1 + (n-nFFTWin/2)/hop
}

// step decides what happens at seek: skip silence, move to where sound
// starts, or decode the window here. samples[0] is absolute sample base.
// It returns the next seek and whether it decoded.
func (s *session) step(ctx context.Context, samples []float32, base, seek, seekEnd int) (int, []Segment, error) {
	rel := seek - base/hop // frame index into samples
	end := min(seek+winFrames, seekEnd)
	v := voiced(samples, rel, rel+(end-seek))
	if countTrue(v) < gateMinVoice {
		return end, nil, nil
	}
	if f := firstSound(v); f > 100 {
		return seek + f - 50, nil, nil
	}
	segs, delta, err := s.window(ctx, samples, base, seek, seekEnd)
	if err != nil {
		return seek, nil, err
	}
	// Drop segments over silence: text there is invented.
	var keep []Segment
	for _, sg := range segs {
		a := int(sg.Start/(10*time.Millisecond)) - seek - 20
		b := int(sg.End/(10*time.Millisecond)) - seek + 20
		a, b = max(a, 0), min(b, len(v))
		if a < b && countTrue(v[a:b]) > 0 {
			keep = append(keep, sg)
		}
	}
	return seek + delta, keep, nil
}

// window decodes the 30 seconds at seek and returns its segments and how
// far to move on (frames).
func (s *session) window(ctx context.Context, samples []float32, base, seek, seekEnd int) ([]Segment, int, error) {
	m := s.m
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, 0, ErrClosed
	}
	v := m.voc
	mel := m.logMel(s.ws, samples, seek*hop-base, s.threads)
	if err := m.encode(ctx, s.ws, mel, s.threads); err != nil {
		return nil, 0, err
	}
	if !s.langSet {
		s.detectLanguage()
	}
	if seek > 0 && seek+500 >= seekEnd {
		s.past = nil // a short last stretch: earlier text tends to make it repeat
	}
	nMax := m.hp.nTextCtx/2 - 4
	maxPrompt := m.hp.nTextCtx / 2

	var best *seq
	var noSpeech float32
	var prompt []int
	for ti, temp := range temperatures {
		nDec := 1
		if temp > 0 {
			nDec = bestOf
		}
		for len(s.seqs) < nDec {
			s.seqs = append(s.seqs, &seq{cache: m.newCache()})
		}
		seqs := s.seqs[:nDec]
		prompt = prompt[:0]
		if temp < condTempCutoff && len(s.past) > 0 {
			take := min(maxPrompt-1, len(s.past))
			prompt = append(prompt, v.prev)
			prompt = append(prompt, s.past[len(s.past)-take:]...)
		}
		prompt = append(prompt, s.init...)
		sotAt := len(prompt) - len(s.init)
		first := seqs[0]
		first.cache.n = 0
		lg := m.decode(s.ws, first.cache, prompt, []int{sotAt, len(prompt) - 1}, s.threads)
		if lg == nil {
			return nil, 0, errors.New("whisper: prompt too long")
		}
		{
			p := append([]float32(nil), lg[0]...)
			softmax(p)
			noSpeech = p[v.nosp]
		}
		for j, q := range seqs {
			*q = seq{cache: q.cache, seekDelta: winFrames}
			if j > 0 {
				q.cache.copyFrom(first.cache, m.hp.nTextState)
			}
			q.logits = append(q.logits[:0], lg[1]...)
			s.process(q, seek, temp)
		}
		for i := 0; i < nMax; i++ {
			if err := ctx.Err(); err != nil {
				return nil, 0, err
			}
			for _, q := range seqs {
				if q.failed || q.done {
					continue
				}
				t := s.sample(q, temp)
				q.tokens = append(q.tokens, t)
				if t.id > v.beg {
					nd := 2 * (t.id - v.beg)
					if q.hasTS && q.seekDelta > nd && q.resultLen < i {
						q.failed = true
						continue
					}
					q.seekDelta, q.resultLen, q.hasTS = nd, i+1, true
				}
				if t.id == v.eot || (q.hasTS && seek+q.seekDelta+deltaMin >= seekEnd) {
					if q.resultLen == 0 {
						if seek+q.seekDelta+deltaMin >= seekEnd {
							q.resultLen = i + 1
						} else {
							q.failed = true
							continue
						}
					}
					q.done = true
					continue
				}
				if i == nMax-1 && (q.resultLen == 0 || q.seekDelta < winFrames/2) {
					q.failed = true // stuck repeating itself
				}
			}
			active := false
			for _, q := range seqs {
				if !q.failed && !q.done {
					active = true
					l := m.decode(s.ws, q.cache, []int{q.tokens[len(q.tokens)-1].id}, []int{0}, s.threads)
					if l == nil {
						q.failed = true
						continue
					}
					q.logits = append(q.logits[:0], l[0]...)
					s.process(q, seek, temp)
				}
			}
			if !active {
				break
			}
			if s.opt.OnProgress != nil && i%16 == 15 {
				s.opt.OnProgress(time.Duration(seek)*10*time.Millisecond, time.Duration(seekEnd)*10*time.Millisecond)
			}
		}
		best = nil
		bestScore := math.Inf(-1)
		for _, q := range seqs {
			if q.failed {
				continue
			}
			q.tokens = q.tokens[:q.resultLen]
			score(q)
			if q.resultLen > 32 && q.en < entropyThold {
				q.failed = true
				continue
			}
			if q.score > bestScore {
				best, bestScore = q, q.score
			}
		}
		if best == nil {
			best = seqs[0]
		}
		if ti == len(temperatures)-1 {
			break
		}
		if !best.failed && !(best.avgLogprob < logprobThold && float64(noSpeech) < noSpeechThold) {
			break
		}
	}

	noSpeechHere := float64(noSpeech) > noSpeechThold && best.avgLogprob < logprobThold
	// The next window's context: what this one was given and what it said.
	var past []int
	if len(prompt) > 0 && prompt[0] == v.prev {
		past = append(past, prompt[1:len(prompt)-len(s.init)]...)
	}
	if !noSpeechHere && !best.failed {
		for _, t := range best.tokens[:best.resultLen] {
			past = append(past, t.id)
		}
	}
	s.past = past

	delta := best.seekDelta
	if best.failed {
		delta = winFrames // give up on this window rather than loop on it
		return nil, min(delta, seekEnd-seek), nil
	}
	var segs []Segment
	if noSpeechHere || len(best.tokens) == 0 {
		return nil, delta, nil
	}
	toks := best.tokens
	t0 := seek + 2*(toks[0].tid-v.beg)
	var text strings.Builder
	emit := func(a, b int) {
		if t := strings.TrimSpace(text.String()); t != "" {
			if b < a {
				b = a
			}
			segs = append(segs, Segment{Start: time.Duration(a) * 10 * time.Millisecond, End: time.Duration(b) * 10 * time.Millisecond, Text: t})
		}
		text.Reset()
	}
	for i := 0; i < len(toks); i++ {
		if toks[i].id < v.eot {
			text.WriteString(v.text(toks[i].id))
		}
		if toks[i].id > v.beg {
			t1 := seek + 2*(toks[i].tid-v.beg)
			emit(t0, t1)
			t0 = t1
			for i+1 < len(toks) && toks[i+1].id > v.beg {
				i++
				t0 = seek + 2*(toks[i].tid-v.beg)
			}
		}
	}
	emit(t0, seek+delta)
	return segs, delta, nil
}

func score(q *seq) {
	if q.resultLen == 0 {
		q.avgLogprob, q.score = math.Inf(-1), math.Inf(-1)
		return
	}
	var sum float64
	for _, t := range q.tokens[:q.resultLen] {
		sum += float64(t.plog)
	}
	q.avgLogprob = sum / float64(q.resultLen)
	q.score = q.avgLogprob
	// Entropy of the last 32 tokens: low means it's going round in circles.
	counts := map[int]int{}
	n := 0
	for _, t := range q.tokens[max(0, q.resultLen-32):q.resultLen] {
		counts[t.id]++
		n++
	}
	var e float64
	for _, c := range counts {
		p := float64(c) / float64(n)
		e -= p * math.Log(p)
	}
	q.en = e
}

var negInf = float32(math.Inf(-1))

// process applies Whisper's rules to the logits for the next token and
// works out the probabilities to sample from.
func (s *session) process(q *seq, seek int, temp float32) {
	v := s.m.voc
	lg := q.logits
	n := len(lg)
	if temp > 0 {
		for i := range lg {
			lg[i] /= temp
		}
	}
	initial := len(q.tokens) == 0
	if initial {
		lg[v.eot] = negInf
		if id, ok := v.toID[" "]; ok {
			lg[id] = negInf
		}
	}
	lg[v.not] = negInf
	lg[v.sot], lg[v.nosp], lg[v.solm] = negInf, negInf, negInf
	lg[v.translate], lg[v.transcribe], lg[v.prev] = negInf, negInf, negInf
	for i := 0; i < v.nLang; i++ {
		lg[v.sot+1+i] = negInf
	}
	for _, t := range s.suppress {
		lg[t] = negInf
	}
	// Timestamps come in pairs, except right before the end.
	nt := len(q.tokens)
	last := nt > 0 && q.tokens[nt-1].id >= v.beg
	penult := nt < 2 || q.tokens[nt-2].id >= v.beg
	if last {
		if penult {
			for i := v.beg; i < n; i++ {
				lg[i] = negInf
			}
		} else {
			for i := 0; i < v.eot; i++ {
				lg[i] = negInf
			}
		}
	}
	if initial {
		prec := 30.0 / float64(s.m.hp.nAudioCtx)
		tid0 := int(math.Round(maxInitialTS / prec))
		for i := v.beg + tid0 + 1; i < n; i++ {
			lg[i] = negInf
		}
	}
	if q.hasTS {
		for i := v.beg; i < v.beg+q.seekDelta/2 && i < n; i++ {
			lg[i] = negInf
		}
	}
	if len(q.logprobs) != n {
		q.logprobs = make([]float32, n)
		q.probs = make([]float32, n)
	}
	logSoftmax(q.logprobs, lg)
	// If the timestamps together outweigh any single text token, it's a timestamp.
	tsMax := negInf
	for _, x := range q.logprobs[v.beg:] {
		tsMax = max(tsMax, x)
	}
	if tsMax > negInf {
		var sum float32
		for _, x := range q.logprobs[v.beg:] {
			if x > negInf {
				sum += expf(x - tsMax)
			}
		}
		tsLP := float32(math.Log(float64(sum))) + tsMax
		textMax := negInf
		for _, x := range q.logprobs[:v.beg] {
			textMax = max(textMax, x)
		}
		if tsLP > textMax {
			for i := 0; i < v.beg; i++ {
				lg[i], q.logprobs[i] = negInf, negInf
			}
		}
	}
	copy(q.probs, lg)
	softmaxInf(q.probs)
}

// sample picks the next token: the most likely at temperature 0, else a draw.
func (s *session) sample(q *seq, temp float32) tok {
	v := s.m.voc
	p := q.probs
	id := 0
	if temp == 0 {
		for i, x := range p {
			if x > p[id] {
				id = i
			}
		}
	} else {
		r := s.rng.Float32()
		var acc float32
		id = -1
		for i, x := range p {
			acc += x
			if r < acc {
				id = i
				break
			}
		}
		if id < 0 {
			for i, x := range p {
				if x > 0 {
					id = i
				}
			}
		}
	}
	tid := v.beg
	for i := v.beg; i < len(p); i++ {
		if p[i] > p[tid] {
			tid = i
		}
	}
	if id >= v.beg {
		tid = id
	}
	return tok{id: id, tid: tid, p: p[id], plog: q.logprobs[id]}
}

func logSoftmax(dst, x []float32) {
	mx := negInf
	for _, v := range x {
		mx = max(mx, v)
	}
	var s float32
	for _, v := range x {
		if v > negInf {
			s += expf(v - mx)
		}
	}
	ls := float32(math.Log(float64(s))) + mx
	for i, v := range x {
		if v > negInf {
			dst[i] = v - ls
		} else {
			dst[i] = negInf
		}
	}
}

// softmaxInf is softmax that leaves -Inf entries at zero.
func softmaxInf(x []float32) {
	mx := negInf
	for _, v := range x {
		mx = max(mx, v)
	}
	var s float32
	for i, v := range x {
		if v > negInf {
			e := expf(v - mx)
			x[i] = e
			s += e
		} else {
			x[i] = 0
		}
	}
	for i := range x {
		x[i] /= s
	}
}

// task is the decoder's task token: translate to English when asked and
// the model can, else transcribe.
func (s *session) task() int {
	if s.opt.Translate && s.m.voc.multilingual {
		return s.m.voc.translate
	}
	return s.m.voc.transcribe
}

// detectLanguage picks the most likely language from the current window.
func (s *session) detectLanguage() {
	m, v := s.m, s.m.voc
	c := m.newCache()
	lg := m.decode(s.ws, c, []int{v.sot}, []int{0}, s.threads)
	best := v.sot + 1
	for i := 0; i < v.nLang; i++ {
		if lg[0][v.sot+1+i] > lg[0][best] {
			best = v.sot + 1 + i
		}
	}
	s.init = []int{v.sot, best, s.task()}
	s.langSet = true
}

// Language is the language detected or chosen, after the first window.
func (s *session) language() string {
	if len(s.init) > 1 {
		return langCodes[s.init[1]-s.m.voc.sot-1]
	}
	return "en"
}

// Transcribe turns 16 kHz mono audio into text.
func (m *Model) Transcribe(ctx context.Context, samples []float32, opt Options) ([]Segment, error) {
	s, err := m.newSession(opt)
	if err != nil {
		return nil, err
	}
	seekEnd := frames(len(samples))
	var all []Segment
	for seek := 0; seek+deltaMin < seekEnd; {
		if err := ctx.Err(); err != nil {
			return all, err
		}
		next, segs, err := s.step(ctx, samples, 0, seek, seekEnd)
		if err != nil {
			return all, err
		}
		for _, sg := range segs {
			all = append(all, sg)
			if opt.OnSegment != nil {
				opt.OnSegment(sg)
			}
		}
		if next <= seek {
			next = seek + winFrames // never stand still
		}
		seek = next
		if opt.OnProgress != nil {
			opt.OnProgress(time.Duration(min(seek, seekEnd))*10*time.Millisecond, time.Duration(seekEnd)*10*time.Millisecond)
		}
	}
	return all, nil
}
