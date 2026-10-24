package conformer

import (
	"errors"
	"sort"
)

// MultiTalker transcribes overlapping speakers: NVIDIA's multitalker
// Parakeet streaming model with Sortformer telling it who talks when.
// It follows NVIDIA's own pipeline (NeMo SpeakerTaggedASR,
// perform_parallel_streaming_stt_spk, with the model card's default
// settings): Sortformer runs on each ASR chunk; every speaker active in
// the last two chunks gets its own ASR stream, fed the chunk's mel
// frames with the frames where that speaker is silent zeroed.
type MultiTalker struct {
	asr     *Model
	diar    *Diarizer
	threads int
	g       *streamGeom
	c       *melChunker
	ds      *DiarStream
	recent  [][]float32 // the last speaker chances, newest last
	spk     []*encState // one ASR stream per speaker seen so far
	frames  int
	// GatingChunks is how many chunks back a speaker counts as active.
	GatingChunks int

	onDiar func(preds [][]float32) // tests: each chunk's speaker chances
}

// SpeakerToken is a token from one speaker's stream.
type SpeakerToken struct {
	Speaker int
	Token
}

// maskValue is the log-mel value NVIDIA's pipeline gives frames that
// were exactly zero before masking (mask_features).
const maskValue = -16.6355

// NewMultiTalker pairs a multitalker model with a Sortformer at right
// context R (13 on the model card).
func NewMultiTalker(asr *Model, diar *Diarizer, right, threads int) (*MultiTalker, error) {
	if asr.spk == nil {
		return nil, errors.New("conformer: this ASR model has no speaker kernels")
	}
	g, err := asr.newGeom(right, threads)
	if err != nil {
		return nil, err
	}
	// MultitalkerTranscriptionConfig.init_diar_model: FIFO 188, cache 188
	p := DiarParams{ChunkLen: 6, RightContext: 7, FIFOLen: 188, CacheLen: 188, UpdatePeriod: diar.Defaults.UpdatePeriod}
	ds, err := diar.NewStream(p, threads)
	if err != nil {
		return nil, err
	}
	return &MultiTalker{asr: asr, diar: diar, threads: threads, g: g, c: newMelChunker(asr.Mel, g, asr.HP.Subsampling), ds: ds, GatingChunks: 2}, nil
}

// Write adds 16 kHz mono samples and returns the tokens that became
// final, each with its speaker.
func (mt *MultiTalker) Write(pcm []float32) []SpeakerToken {
	var out []SpeakerToken
	for _, p := range pieces(finite(pcm), mt.asr.Mel.cfg.SampleRate) {
		out = append(out, mt.run(mt.c.write(p))...)
	}
	return out
}

// Close finishes the recording.
func (mt *MultiTalker) Close() []SpeakerToken { return mt.run(mt.c.close()) }

func (mt *MultiTalker) run(chunks []melChunk) []SpeakerToken {
	var out []SpeakerToken
	nm := mt.asr.Mel.cfg.NumMels
	per := mt.g.chunk
	sub := mt.asr.HP.Subsampling
	for _, ch := range chunks {
		nf := len(ch.frames)
		mel := chunkMel(ch.frames, nm)
		drop := 0
		if !ch.first {
			drop = mt.g.drop
		}
		preds := mt.ds.step(mel, nf, ch.length, drop, 0, 0)
		if mt.onDiar != nil {
			mt.onDiar(preds)
		}
		mt.recent = append(mt.recent, preds...)
		if keep := per * mt.GatingChunks; len(mt.recent) > keep {
			mt.recent = append([][]float32(nil), mt.recent[len(mt.recent)-keep:]...)
		}
		chunkPreds := mt.recent[max(0, len(mt.recent)-per):]
		active := mt.activeSpeakers()
		nFrames := 0
		for _, spk := range active {
			for len(mt.spk) <= spk {
				mt.spk = append(mt.spk, mt.asr.newEncState())
			}
			// mask_features: the speaker's chunk chances > 0.5, 8 mel
			// frames each, aligned to the chunk's end
			mask := make([]float32, 0, len(chunkPreds)*sub)
			for _, p := range chunkPreds {
				v := float32(0)
				if p[spk] > 0.5 {
					v = 1
				}
				for i := 0; i < sub; i++ {
					mask = append(mask, v)
				}
			}
			if len(mask) > nf {
				mask = mask[:nf]
			} else if len(mask) < nf {
				mask = append(make([]float32, nf-len(mask)), mask...)
			}
			masked := make([]float32, len(mel))
			for j := 0; j < nm; j++ {
				for t := 0; t < nf; t++ {
					v := mel[j*nf+t]
					switch {
					case v == 0:
						masked[j*nf+t] = maskValue
					default:
						masked[j*nf+t] = v * mask[t]
					}
				}
			}
			st := mt.spk[spk]
			x, L := mt.asr.streamStep(mt.g, st, masked, nf, ch.length, ch.first, mt.threads)
			nFrames = max(nFrames, L)
			if L == 0 {
				continue
			}
			for _, t := range mt.asr.Decode(x, L, st.dec, mt.frames, mt.threads) {
				out = append(out, SpeakerToken{Speaker: spk, Token: t})
			}
		}
		if nFrames == 0 {
			nFrames = len(preds)
		}
		mt.frames += nFrames
	}
	return out
}

// activeSpeakers is NeMo's _find_active_speakers with cache gating:
// speakers whose chance passed 0.5 in the last GatingChunks chunks.
func (mt *MultiTalker) activeSpeakers() []int {
	ns := mt.diar.Speakers
	var act []int
	for j := 0; j < ns; j++ {
		var mx float32
		for _, p := range mt.recent {
			mx = max(mx, p[j])
		}
		if mx > 0.5 {
			act = append(act, j)
		}
	}
	sort.Ints(act)
	return act
}

// SpeakerText joins each speaker's tokens into text.
func (mt *MultiTalker) SpeakerText(toks []SpeakerToken) map[int]string {
	by := map[int][]int{}
	for _, t := range toks {
		by[t.Speaker] = append(by[t.Speaker], t.ID)
	}
	out := map[int]string{}
	for s, ids := range by {
		out[s] = mt.asr.Text(ids)
	}
	return out
}
