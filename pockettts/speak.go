package pockettts

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
)

// Options tune Speak. The zero value takes the model's own settings.
type Options struct {
	// Temperature scales the noise each frame starts from (its variance);
	// 0 means the model's, 0.3.
	Temperature float32
	// Seed makes the output repeatable: the same text, voice, options and
	// seed give the same samples on the same kind of processor.
	Seed uint64
	// DecodeSteps is the flow head's number of steps; 0 means the
	// model's, 1.
	DecodeSteps int
	// EOSThreshold is the end-of-speech logit above which a chunk ends;
	// 0 means the model's, -4.
	EOSThreshold float32
	// FramesAfterEOS is how many 80 ms frames follow the end of speech;
	// 0 lets the model guess from the chunk's length (3 for 5 words or
	// more, 5 for fewer).
	FramesAfterEOS int
	// Threads is how many goroutines the matrix products use; 0 means
	// GOMAXPROCS.
	Threads int

	// tests only
	noise func(dst []float32)                                            // in place of the random noise
	trace func(lmOut, noise, latent []float32, eos float32, prompt bool) // every flow head call
}

// ErrStop can be returned by an emit function to end Speak early
// without an error.
var ErrStop = errors.New("pockettts: stopped")

// Speak reads text aloud in voice v. Long text is split into chunks of
// whole sentences, each spoken from the voice's state. emit gets each
// frame's audio, FrameSamples() samples at SampleRate(), as soon as it
// is made, on the caller's goroutine; it may keep the slice. If emit
// returns an error, Speak stops and returns it (nil for ErrStop).
func (m *Model) Speak(text string, v *Voice, opt Options, emit func(pcm []float32) error) error {
	if v == nil || v.model != m.sha {
		return errors.New("pockettts: the voice was made for a different model")
	}
	o := m.fill(opt)
	chunks, err := m.chunks(text, m.cfg.maxTokens)
	if err != nil {
		return err
	}
	rng := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))
	tc := m.lm.head.timeConds(o.DecodeSteps)
	for _, c := range chunks {
		prep, guess, err := m.cfg.prepareText(c)
		if err != nil {
			return err
		}
		after := o.FramesAfterEOS
		if opt.FramesAfterEOS == 0 && m.cfg.framesAfterEOS < 0 {
			after = guess + 2
		}
		if err := m.speakChunk(m.tok.encode(prep), v, o, after, tc, rng, emit); err != nil {
			if err == ErrStop {
				return nil
			}
			return err
		}
	}
	return nil
}

// Synthesize returns all of Speak's audio.
func (m *Model) Synthesize(text string, v *Voice, opt Options) ([]float32, error) {
	var out []float32
	err := m.Speak(text, v, opt, func(p []float32) error {
		out = append(out, p...)
		return nil
	})
	return out, err
}

func (m *Model) fill(o Options) Options {
	if o.Temperature == 0 {
		o.Temperature = m.cfg.temperature
	}
	if o.DecodeSteps <= 0 {
		o.DecodeSteps = m.cfg.decodeSteps
	}
	if o.EOSThreshold == 0 {
		o.EOSThreshold = m.cfg.eosThreshold
	}
	if o.FramesAfterEOS <= 0 {
		o.FramesAfterEOS = max(0, m.cfg.framesAfterEOS)
	}
	if o.Threads <= 0 {
		o.Threads = runtime.GOMAXPROCS(0)
	}
	return o
}

// maxFrames is _estimate_max_gen_len: 3 tokens a second plus 2 s.
func (m *Model) maxFrames(tokens int) int {
	fr := float64(m.cfg.sampleRate) / float64(m.cfg.frameSamples)
	return int(math.Ceil((float64(tokens)/3 + 2) * fr))
}

// speakChunk is _generate_audio_stream_short_text for one chunk. As in
// Kyutai's code, the flow LM makes latents on a goroutine of its own
// while the caller's goroutine decodes them and calls emit, so the two
// overlap.
func (m *Model) speakChunk(ids []int, v *Voice, o Options, after int, tc [][]float32, rng *rand.Rand, emit func([]float32) error) error {
	st := v.st.clone((len(ids) + m.maxFrames(len(ids))) * m.lm.tf.d)
	ws := &scratch{}
	out := m.lm.text(ids, st, ws, o.Threads)
	if o.trace != nil {
		o.trace(out, nil, nil, 0, true)
	}
	lat := make(chan []float32, 8)
	done := make(chan struct{})
	stopped := false
	stop := func() {
		if !stopped {
			stopped = true
			close(done)
		}
	}
	defer stop()
	var lmErr error // written before lat closes
	go func() {
		defer close(lat)
		defer func() {
			if r := recover(); r != nil {
				lmErr = fmt.Errorf("pockettts: %v", r)
			}
		}()
		ldim := m.lm.head.ldim
		std := float32(math.Sqrt(float64(o.Temperature)))
		noise := make([]float32, ldim)
		var latent []float32
		eosAt := -1
		for step := 0; step < m.maxFrames(len(ids)); step++ {
			h, eos := m.lm.step(latent, st, ws, o.Threads)
			if o.noise != nil {
				o.noise(noise)
			} else {
				for i := range noise {
					noise[i] = float32(rng.NormFloat64()) * std
				}
			}
			next := m.lm.head.sample(h, noise, tc, ws)
			if o.trace != nil {
				o.trace(h, noise, next, eos, false)
			}
			if eos > o.EOSThreshold && eosAt < 0 && step >= m.cfg.minFramesBeforeEOS {
				eosAt = step
			}
			if eosAt >= 0 && step >= eosAt+after {
				return
			}
			scaled := make([]float32, ldim)
			for i, x := range next {
				scaled[i] = x*m.lm.std[i] + m.lm.mean[i]
			}
			select {
			case lat <- scaled:
			case <-done:
				return
			}
			latent = next
		}
	}()
	dst := m.dec.newState()
	fade := m.cfg.sampleRate / 200
	var err error
	for scaled := range lat {
		if err != nil {
			continue // let the flow LM see done and finish
		}
		frame := append([]float32(nil), m.dec.decode(dst, scaled, 1, o.Threads)...)
		if !dst.started {
			for i := 0; i < min(fade, len(frame)); i++ {
				frame[i] *= float32(i) / float32(fade-1)
			}
			dst.started = true
		}
		if err = emit(frame); err != nil {
			stop()
		}
	}
	if err != nil {
		return err
	}
	return lmErr
}
