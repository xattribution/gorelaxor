//go:build !windows

package whisper

import "errors"

func platformCanDecode(ext string) bool { return false }

func platformDecode(path string) ([]float32, int, error) {
	return nil, 0, errors.New("only WAV audio can be read on this computer")
}
