//go:build windows

package whisper

import (
	"errors"
	"fmt"
	"runtime"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// Media Foundation decodes the audio formats Windows itself plays (MP3,
// AAC in M4A and MP4, WMA, video files' sound tracks). It's called through
// its COM interfaces directly, so no cgo and nothing extra to install.

var (
	mfplat       = windows.NewLazySystemDLL("mfplat.dll")
	mfreadwrite  = windows.NewLazySystemDLL("mfreadwrite.dll")
	ole32        = windows.NewLazySystemDLL("ole32.dll")
	procStartup  = mfplat.NewProc("MFStartup")
	procShutdown = mfplat.NewProc("MFShutdown")
	procNewType  = mfplat.NewProc("MFCreateMediaType")
	procReader   = mfreadwrite.NewProc("MFCreateSourceReaderFromURL")
	procCoInit   = ole32.NewProc("CoInitializeEx")
	procCoUninit = ole32.NewProc("CoUninitialize")
)

var (
	mfMTMajorType  = windows.GUID{Data1: 0x48eba18e, Data2: 0xf8c9, Data3: 0x4687, Data4: [8]byte{0xbf, 0x11, 0x0a, 0x74, 0xc9, 0xf9, 0x6a, 0x8f}}
	mfMTSubtype    = windows.GUID{Data1: 0xf7e34c9a, Data2: 0x42e8, Data3: 0x4714, Data4: [8]byte{0xb7, 0x4b, 0xcb, 0x29, 0xd7, 0x2c, 0x35, 0xe5}}
	mfMTChannels   = windows.GUID{Data1: 0x37e48bf5, Data2: 0x645e, Data3: 0x4c5b, Data4: [8]byte{0x89, 0xde, 0xad, 0xa9, 0xe2, 0x9b, 0x69, 0x6a}}
	mfMTRate       = windows.GUID{Data1: 0x5faeeae7, Data2: 0x0290, Data3: 0x4c31, Data4: [8]byte{0x9e, 0x8a, 0xc5, 0x34, 0xf6, 0x8d, 0x9d, 0xba}}
	mfMTBits       = windows.GUID{Data1: 0xf2deb57f, Data2: 0x40fa, Data3: 0x4764, Data4: [8]byte{0xaa, 0x33, 0xed, 0x4f, 0x2d, 0x1f, 0xf6, 0x69}}
	mfMediaAudio   = windows.GUID{Data1: 0x73647561, Data2: 0x0000, Data3: 0x0010, Data4: [8]byte{0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}}
	mfAudioFloat   = windows.GUID{Data1: 0x00000003, Data2: 0x0000, Data3: 0x0010, Data4: [8]byte{0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}}
	mfAudioPCM     = windows.GUID{Data1: 0x00000001, Data2: 0x0000, Data3: 0x0010, Data4: [8]byte{0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}}
	mfFirstAudio   = uintptr(0xFFFFFFFD)
	mfAllStreams   = uintptr(0xFFFFFFFE)
	mfReaderError  = uint32(0x1)
	mfReaderEOS    = uint32(0x2)
	mfReaderNewFmt = uint32(0x20)
)

// Vtable slots (IUnknown is 0-2; IMFAttributes 3-32).
const (
	vRelease          = 2
	vGetUINT32        = 7
	vSetUINT32        = 21
	vSetGUID          = 24
	vSetStreamSel     = 4
	vGetCurrentType   = 6
	vSetCurrentType   = 7
	vReadSample       = 9
	vContiguousBuffer = 41
	vLock             = 3
	vUnlock           = 4
)

func call(obj unsafe.Pointer, slot int, args ...uintptr) uintptr {
	vtbl := *(*unsafe.Pointer)(obj)
	fn := *(*uintptr)(unsafe.Add(vtbl, uintptr(slot)*unsafe.Sizeof(uintptr(0))))
	r, _, _ := syscall.SyscallN(fn, append([]uintptr{uintptr(obj)}, args...)...)
	return r
}

func release(obj unsafe.Pointer) {
	if obj != nil {
		call(obj, vRelease)
	}
}

func hr(r uintptr, what string) error {
	if int32(r) < 0 {
		return fmt.Errorf("Windows couldn't read the audio (%s, 0x%08X)", what, uint32(r))
	}
	return nil
}

var mfExts = map[string]bool{".mp3": true, ".m4a": true, ".mp4": true, ".aac": true, ".wma": true, ".wmv": true,
	".wav": true, ".mov": true, ".3gp": true, ".3g2": true, ".m4v": true, ".asf": true, ".adts": true, ".flac": true}

func platformCanDecode(ext string) bool {
	return mfExts[ext] && mfplat.Load() == nil && mfreadwrite.Load() == nil
}

// platformDecode reads the file's first sound track as mono float samples
// at its own rate.
func platformDecode(path string) ([]float32, int, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if r, _, _ := procCoInit.Call(0, 0); int32(r) >= 0 { // COINIT_MULTITHREADED
		defer procCoUninit.Call()
	}
	if r, _, _ := procStartup.Call(0x00020070, 0); int32(r) < 0 {
		if r, _, _ = procStartup.Call(0x00010070, 0); int32(r) < 0 {
			return nil, 0, hr(r, "MFStartup")
		}
	}
	defer procShutdown.Call()

	url, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return nil, 0, err
	}
	var reader unsafe.Pointer
	r, _, _ := procReader.Call(uintptr(unsafe.Pointer(url)), 0, uintptr(unsafe.Pointer(&reader)))
	if err := hr(r, "open"); err != nil {
		return nil, 0, err
	}
	defer release(reader)
	call(reader, vSetStreamSel, mfAllStreams, 0)
	if err := hr(call(reader, vSetStreamSel, mfFirstAudio, 1), "no sound track"); err != nil {
		return nil, 0, err
	}
	float := true
	if err := setOutput(reader, true); err != nil {
		float = false
		if err := setOutput(reader, false); err != nil {
			return nil, 0, err
		}
	}
	ch, rate, err := outputFormat(reader)
	if err != nil {
		return nil, 0, err
	}
	var out []float32
	for {
		var idx, flags uint32
		var ts int64
		var sample unsafe.Pointer
		r := call(reader, vReadSample, mfFirstAudio, 0, uintptr(unsafe.Pointer(&idx)), uintptr(unsafe.Pointer(&flags)), uintptr(unsafe.Pointer(&ts)), uintptr(unsafe.Pointer(&sample)))
		if err := hr(r, "read"); err != nil {
			return nil, 0, err
		}
		if flags&mfReaderError != 0 {
			release(sample)
			return nil, 0, errors.New("Windows couldn't decode the audio")
		}
		if flags&mfReaderNewFmt != 0 {
			if ch, rate, err = outputFormat(reader); err != nil {
				release(sample)
				return nil, 0, err
			}
		}
		if sample != nil {
			var buf unsafe.Pointer
			if hr(call(sample, vContiguousBuffer, uintptr(unsafe.Pointer(&buf))), "buffer") == nil {
				var p unsafe.Pointer
				var max, cur uint32
				if hr(call(buf, vLock, uintptr(unsafe.Pointer(&p)), uintptr(unsafe.Pointer(&max)), uintptr(unsafe.Pointer(&cur))), "lock") == nil {
					out = appendMono(out, p, int(cur), ch, float)
					call(buf, vUnlock)
					if rate > 0 && len(out) > MaxSeconds*rate {
						release(buf)
						release(sample)
						return nil, 0, errTooLong
					}
				}
				release(buf)
			}
			release(sample)
		}
		if flags&mfReaderEOS != 0 {
			break
		}
	}
	return out, rate, nil
}

func setOutput(reader unsafe.Pointer, float bool) error {
	var mt unsafe.Pointer
	r, _, _ := procNewType.Call(uintptr(unsafe.Pointer(&mt)))
	if err := hr(r, "media type"); err != nil {
		return err
	}
	defer release(mt)
	sub := &mfAudioPCM
	if float {
		sub = &mfAudioFloat
	}
	call(mt, vSetGUID, uintptr(unsafe.Pointer(&mfMTMajorType)), uintptr(unsafe.Pointer(&mfMediaAudio)))
	call(mt, vSetGUID, uintptr(unsafe.Pointer(&mfMTSubtype)), uintptr(unsafe.Pointer(sub)))
	if !float {
		call(mt, vSetUINT32, uintptr(unsafe.Pointer(&mfMTBits)), 16)
	}
	return hr(call(reader, vSetCurrentType, mfFirstAudio, 0, uintptr(mt)), "output format")
}

func outputFormat(reader unsafe.Pointer) (ch, rate int, err error) {
	var mt unsafe.Pointer
	if err := hr(call(reader, vGetCurrentType, mfFirstAudio, uintptr(unsafe.Pointer(&mt))), "format"); err != nil {
		return 0, 0, err
	}
	defer release(mt)
	var c, r uint32
	call(mt, vGetUINT32, uintptr(unsafe.Pointer(&mfMTChannels)), uintptr(unsafe.Pointer(&c)))
	call(mt, vGetUINT32, uintptr(unsafe.Pointer(&mfMTRate)), uintptr(unsafe.Pointer(&r)))
	if c == 0 || c > 32 || r < 1000 {
		return 0, 0, errors.New("Windows gave the audio in a format this program doesn't expect")
	}
	return int(c), int(r), nil
}

// appendMono averages interleaved frames at p (float32 or int16) to mono.
func appendMono(out []float32, p unsafe.Pointer, n, ch int, float bool) []float32 {
	if float {
		s := unsafe.Slice((*float32)(p), n/4)
		for i := 0; i+ch <= len(s); i += ch {
			var v float32
			for c := 0; c < ch; c++ {
				v += s[i+c]
			}
			out = append(out, v/float32(ch))
		}
		return out
	}
	s := unsafe.Slice((*int16)(p), n/2)
	for i := 0; i+ch <= len(s); i += ch {
		var v float32
		for c := 0; c < ch; c++ {
			v += float32(s[i+c]) / 32768
		}
		out = append(out, v/float32(ch))
	}
	return out
}
