package whisper

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Stream transcribes audio as it arrives, such as a meeting being
// recorded. Each 30-second window is decoded once enough audio for it has
// come in, so text arrives about half a minute behind the speech; Close
// decodes the rest. The result is the same as Transcribe on the whole
// recording.
type Stream struct {
	m    *Model
	ctx  context.Context
	opt  Options
	sess *session
	err0 error

	mu     sync.Mutex
	buf    []float32 // samples from base on
	base   int       // absolute index of buf[0], a multiple of hop
	total  int       // samples written
	closed bool
	wake   chan struct{}
	done   chan struct{}

	seek int
	segs []Segment
	err  error
}

// NewStream starts a live transcription. Write audio to it as it comes
// (16 kHz mono) and Close it at the end.
func (m *Model) NewStream(ctx context.Context, opt Options) *Stream {
	s := &Stream{m: m, ctx: ctx, opt: opt, wake: make(chan struct{}, 1), done: make(chan struct{})}
	s.sess, s.err0 = m.newSession(opt)
	go s.run()
	return s
}

var errStreamClosed = errors.New("whisper: the stream is closed")

// Write adds audio. It copies the samples and returns at once; decoding
// happens in the background.
func (s *Stream) Write(samples []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return errStreamClosed
	}
	if s.err != nil {
		return s.err
	}
	s.buf = append(s.buf, samples...)
	s.total += len(samples)
	select {
	case s.wake <- struct{}{}:
	default:
	}
	return nil
}

// Close decodes whatever is left and returns every segment.
func (s *Stream) Close() ([]Segment, error) {
	s.mu.Lock()
	if !s.closed {
		s.closed = true
		select {
		case s.wake <- struct{}{}:
		default:
		}
	}
	s.mu.Unlock()
	<-s.done
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.segs, s.err
}

func (s *Stream) fail(err error) {
	s.mu.Lock()
	s.err = err
	s.mu.Unlock()
}

func (s *Stream) run() {
	defer close(s.done)
	defer func() {
		if p := recover(); p != nil {
			s.fail(fmt.Errorf("whisper: transcription failed: %v", p))
		}
	}()
	if s.err0 != nil {
		s.fail(s.err0)
		return
	}
	for {
		s.mu.Lock()
		buf, base, total, closed := s.buf, s.base, s.total, s.closed
		s.mu.Unlock()
		if err := s.ctx.Err(); err != nil {
			s.fail(err)
			return
		}
		seekEnd := frames(total)
		// A window can be decoded once audio runs past it, exactly as it
		// would be with the whole recording in hand.
		ready := closed || seekEnd > s.seek+winFrames+deltaMin
		if ready && s.seek+deltaMin < seekEnd {
			next, segs, err := s.sess.step(s.ctx, buf, base, s.seek, seekEnd)
			if err != nil {
				s.fail(err)
				return
			}
			if next <= s.seek {
				next = s.seek + winFrames
			}
			s.seek = next
			func() {
				s.mu.Lock()
				defer s.mu.Unlock() // a panic here must not leave it locked for fail
				s.segs = append(s.segs, segs...)
				// Keep a frame of audio before the next window; drop the rest.
				if keep := s.seek*hop - nFFTWin - s.base; keep > 0 {
					// A short stream closed early can seek past its audio.
					keep = min(keep, len(s.buf))
					keep -= keep % hop
					s.buf = append([]float32(nil), s.buf[keep:]...)
					s.base += keep
				}
			}()
			for _, sg := range segs {
				if s.opt.OnSegment != nil {
					s.opt.OnSegment(sg)
				}
			}
			if s.opt.OnProgress != nil {
				s.opt.OnProgress(time.Duration(min(s.seek, seekEnd))*10*time.Millisecond, time.Duration(seekEnd)*10*time.Millisecond)
			}
			continue
		}
		if closed {
			return
		}
		select {
		case <-s.wake:
		case <-s.ctx.Done():
		}
	}
}
