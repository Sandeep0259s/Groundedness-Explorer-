import av
import numpy as np

from src.rag.ingest import VIDEO_SUFFIXES, _extract_keyframes


def _make_test_video(path, seconds=3, fps=5, size=(160, 120)):
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height = size
    stream.pix_fmt = "yuv420p"
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200)]
    for sec in range(seconds):
        color = colors[sec % len(colors)]
        for _ in range(fps):
            arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            arr[:, :] = color
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_extract_keyframes_returns_requested_count(tmp_path):
    video_path = tmp_path / "test.mp4"
    _make_test_video(video_path)

    frames = _extract_keyframes(video_path, count=3)
    assert len(frames) == 3
    for frame_bytes in frames:
        assert frame_bytes[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_extract_keyframes_handles_corrupt_file_gracefully(tmp_path):
    bad_path = tmp_path / "not_a_video.mp4"
    bad_path.write_bytes(b"this is not a real video file")
    assert _extract_keyframes(bad_path) == []


def test_video_suffixes_exclude_audio_only_formats():
    assert ".mp4" in VIDEO_SUFFIXES
    assert ".mp3" not in VIDEO_SUFFIXES
    assert ".wav" not in VIDEO_SUFFIXES
