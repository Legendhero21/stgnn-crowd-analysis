"""
Video Source Abstraction
------------------------
Production-grade video input handling for edge devices.

Supports:
- Local video files (mp4, avi, etc.)
- RTSP streams
- HTTP streams
- Webcam input

Thread-safe with graceful error handling.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterator, Callable

import cv2
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """
    Container for a single video frame with metadata.
    
    Attributes:
        frame: BGR image array (H, W, 3), np.uint8.
        frame_idx: Zero-based frame index.
        timestamp_ms: Timestamp in milliseconds from source.
        source_id: Identifier of the video source.
    """
    frame: np.ndarray
    frame_idx: int
    timestamp_ms: float
    source_id: str
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return (H, W, C)."""
        return self.frame.shape
    
    @property
    def height(self) -> int:
        return self.frame.shape[0]
    
    @property
    def width(self) -> int:
        return self.frame.shape[1]
    
    def is_valid(self) -> bool:
        """Check if frame data is valid."""
        return (
            self.frame is not None
            and self.frame.size > 0
            and self.frame.ndim == 3
            and self.frame.shape[2] == 3
        )


class VideoSource(ABC):
    """
    Abstract base class for video sources.
    
    All video sources must implement:
    - open(): Initialize the source
    - read(): Get the next frame
    - close(): Release resources
    - is_open(): Check if source is available
    
    Usage:
        with VideoFileSource("video.mp4") as source:
            for frame_data in source:
                process(frame_data.frame)
    """
    
    @abstractmethod
    def open(self) -> bool:
        """
        Open the video source.
        
        Returns:
            True if source opened successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def read(self) -> Optional[FrameData]:
        """
        Read the next frame.
        
        Returns:
            FrameData if successful, None if end of stream or error.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Release video source resources."""
        pass
    
    @abstractmethod
    def is_open(self) -> bool:
        """Check if source is currently open and readable."""
        pass
    
    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source."""
        pass
    
    @property
    @abstractmethod
    def fps(self) -> float:
        """Frames per second of the source."""
        pass
    
    @property
    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        """Frame dimensions as (width, height)."""
        pass
    
    @property
    @abstractmethod
    def total_frames(self) -> Optional[int]:
        """Total frame count (None for live streams)."""
        pass
    
    def __enter__(self) -> "VideoSource":
        """Context manager entry."""
        if not self.open():
            raise RuntimeError(f"Failed to open video source: {self.source_id}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __iter__(self) -> Iterator[FrameData]:
        """Iterate over frames."""
        while self.is_open():
            frame_data = self.read()
            if frame_data is None:
                break
            yield frame_data


class VideoFileSource(VideoSource):
    """
    Video source from a local file.
    
    Supports looping for continuous operation during testing.
    """
    
    def __init__(
        self,
        file_path: str,
        loop: bool = False,
        max_loops: Optional[int] = None,
    ):
        """
        Initialize video file source.
        
        Args:
            file_path: Path to video file.
            loop: If True, restart from beginning when end is reached.
            max_loops: Maximum number of loops (None = infinite if loop=True).
        
        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        self._file_path = Path(file_path)
        if not self._file_path.is_file():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        self._loop = loop
        self._max_loops = max_loops
        self._loop_count = 0
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx = 0
        self._lock = threading.Lock()
        
        self._fps: float = 0.0
        self._frame_width: int = 0
        self._frame_height: int = 0
        self._total_frames: int = 0
    
    @property
    def source_id(self) -> str:
        return str(self._file_path)
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self._frame_width, self._frame_height)
    
    @property
    def total_frames(self) -> Optional[int]:
        return self._total_frames if self._total_frames > 0 else None
    
    @property
    def loop_count(self) -> int:
        """Number of times the video has looped."""
        return self._loop_count
    
    def open(self) -> bool:
        """Open the video file."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
            
            self._cap = cv2.VideoCapture(str(self._file_path))
            
            if not self._cap.isOpened():
                logger.error("Failed to open video: %s", self._file_path)
                return False
            
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._frame_idx = 0
            self._loop_count = 0
            
            if self._fps <= 0:
                logger.warning("Invalid FPS, defaulting to 30")
                self._fps = 30.0
            
            logger.info(
                "Opened video: %s (%dx%d @ %.1f FPS, %d frames)",
                self._file_path.name,
                self._frame_width,
                self._frame_height,
                self._fps,
                self._total_frames,
            )
            
            return True
    
    def read(self) -> Optional[FrameData]:
        """Read the next frame from the video file."""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            
            ret, frame = self._cap.read()
            
            if not ret:
                # End of video
                if self._loop:
                    if self._max_loops is not None and self._loop_count >= self._max_loops:
                        logger.info("Max loops reached (%d), stopping", self._max_loops)
                        return None
                    
                    # Seek to beginning
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._loop_count += 1
                    self._frame_idx = 0
                    logger.info("Video looped (count: %d)", self._loop_count)
                    
                    ret, frame = self._cap.read()
                    if not ret:
                        logger.error("Failed to read after loop reset")
                        return None
                else:
                    logger.info("End of video reached")
                    return None
            
            timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            
            frame_data = FrameData(
                frame=frame,
                frame_idx=self._frame_idx,
                timestamp_ms=timestamp_ms,
                source_id=self.source_id,
            )
            
            self._frame_idx += 1
            
            return frame_data
    
    def seek(self, frame_idx: int) -> bool:
        """
        Seek to a specific frame.
        
        Args:
            frame_idx: Target frame index.
        
        Returns:
            True if seek succeeded.
        """
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False
            
            if frame_idx < 0 or (self._total_frames > 0 and frame_idx >= self._total_frames):
                return False
            
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self._frame_idx = frame_idx
            return True
    
    def close(self) -> None:
        """Release the video file."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            logger.info("Closed video source: %s", self._file_path.name)
    
    def is_open(self) -> bool:
        """Check if video is open."""
        with self._lock:
            return self._cap is not None and self._cap.isOpened()


class VideoStreamSource(VideoSource):
    """
    Video source from a network stream (RTSP, HTTP, etc.).
    
    Handles reconnection on stream failure.
    """
    
    def __init__(
        self,
        stream_url: str,
        reconnect_attempts: int = 5,
        reconnect_delay_sec: float = 2.0,
        connection_timeout_sec: float = 10.0,
    ):
        """
        Initialize stream source.
        
        Args:
            stream_url: URL of the stream (rtsp://, http://).
            reconnect_attempts: Number of reconnection attempts on failure.
            reconnect_delay_sec: Delay between reconnection attempts.
            connection_timeout_sec: Timeout for initial connection.
        """
        self._stream_url = stream_url
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay_sec = reconnect_delay_sec
        self._connection_timeout_sec = connection_timeout_sec
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx = 0
        self._lock = threading.Lock()
        self._is_running = False
        
        self._fps: float = 30.0  # Default for streams
        self._frame_width: int = 0
        self._frame_height: int = 0
        
        self._last_frame_time: float = 0.0
        self._consecutive_failures: int = 0
    
    @property
    def source_id(self) -> str:
        return self._stream_url
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self._frame_width, self._frame_height)
    
    @property
    def total_frames(self) -> Optional[int]:
        # Live streams don't have a total frame count
        return None
    
    def open(self) -> bool:
        """Open the stream with retry logic."""
        for attempt in range(self._reconnect_attempts):
            logger.info(
                "Connecting to stream: %s (attempt %d/%d)",
                self._stream_url,
                attempt + 1,
                self._reconnect_attempts,
            )
            
            with self._lock:
                if self._cap is not None:
                    self._cap.release()
                
                # OpenCV VideoCapture for streams
                self._cap = cv2.VideoCapture(self._stream_url)
                
                # Set timeout (OpenCV 4.x+)
                try:
                    self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 
                                  int(self._connection_timeout_sec * 1000))
                except Exception:
                    pass  # Property may not be available
                
                if self._cap.isOpened():
                    # Read one frame to verify connection
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
                        if self._fps <= 0:
                            self._fps = 30.0
                        
                        self._frame_width = frame.shape[1]
                        self._frame_height = frame.shape[0]
                        self._frame_idx = 1
                        self._is_running = True
                        self._consecutive_failures = 0
                        
                        logger.info(
                            "Connected to stream: %s (%dx%d @ %.1f FPS)",
                            self._stream_url,
                            self._frame_width,
                            self._frame_height,
                            self._fps,
                        )
                        
                        return True
            
            logger.warning("Connection attempt %d failed", attempt + 1)
            time.sleep(self._reconnect_delay_sec)
        
        logger.error("Failed to connect to stream after %d attempts", self._reconnect_attempts)
        return False
    
    def read(self) -> Optional[FrameData]:
        """Read the next frame from the stream."""
        with self._lock:
            if self._cap is None or not self._is_running:
                return None
            
            ret, frame = self._cap.read()
            
            if not ret or frame is None:
                self._consecutive_failures += 1
                
                if self._consecutive_failures >= 10:
                    logger.warning(
                        "Too many consecutive failures (%d), attempting reconnect",
                        self._consecutive_failures,
                    )
                    self._is_running = False
                    # Caller should handle reconnection
                    return None
                
                return None
            
            self._consecutive_failures = 0
            current_time = time.time()
            
            frame_data = FrameData(
                frame=frame,
                frame_idx=self._frame_idx,
                timestamp_ms=current_time * 1000,
                source_id=self.source_id,
            )
            
            self._frame_idx += 1
            self._last_frame_time = current_time
            
            return frame_data
    
    def close(self) -> None:
        """Close the stream."""
        with self._lock:
            self._is_running = False
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            logger.info("Closed stream source: %s", self._stream_url)
    
    def is_open(self) -> bool:
        """Check if stream is active."""
        with self._lock:
            return self._is_running and self._cap is not None


class WebcamSource(VideoSource):
    """
    Video source from a local webcam.
    """
    
    def __init__(self, device_id: int = 0, fps: float = 30.0):
        """
        Initialize webcam source.
        
        Args:
            device_id: Webcam device index (0 for default).
            fps: Target FPS (may not be honored by hardware).
        """
        self._device_id = device_id
        self._target_fps = fps
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx = 0
        self._lock = threading.Lock()
        
        self._actual_fps: float = fps
        self._frame_width: int = 0
        self._frame_height: int = 0
    
    @property
    def source_id(self) -> str:
        return f"webcam:{self._device_id}"
    
    @property
    def fps(self) -> float:
        return self._actual_fps
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self._frame_width, self._frame_height)
    
    @property
    def total_frames(self) -> Optional[int]:
        return None  # Live source
    
    def open(self) -> bool:
        """Open the webcam."""
        with self._lock:
            self._cap = cv2.VideoCapture(self._device_id)
            
            if not self._cap.isOpened():
                logger.error("Failed to open webcam: %d", self._device_id)
                return False
            
            # Try to set FPS
            self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
            
            self._actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if self._actual_fps <= 0:
                self._actual_fps = self._target_fps
            
            self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._frame_idx = 0
            
            logger.info(
                "Opened webcam: %d (%dx%d @ %.1f FPS)",
                self._device_id,
                self._frame_width,
                self._frame_height,
                self._actual_fps,
            )
            
            return True
    
    def read(self) -> Optional[FrameData]:
        """Read frame from webcam."""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            
            ret, frame = self._cap.read()
            
            if not ret or frame is None:
                return None
            
            frame_data = FrameData(
                frame=frame,
                frame_idx=self._frame_idx,
                timestamp_ms=time.time() * 1000,
                source_id=self.source_id,
            )
            
            self._frame_idx += 1
            
            return frame_data
    
    def close(self) -> None:
        """Release webcam."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            logger.info("Closed webcam: %d", self._device_id)
    
    def is_open(self) -> bool:
        """Check if webcam is open."""
        with self._lock:
            return self._cap is not None and self._cap.isOpened()


def create_video_source(source: str, loop: bool = False) -> VideoSource:
    """
    Factory function to create appropriate video source.
    
    Args:
        source: File path, stream URL, or "webcam:N".
        loop: If True, loop video files.
    
    Returns:
        Appropriate VideoSource instance.
    
    Raises:
        ValueError: If source format is not recognized.
    """
    if source.startswith("webcam:"):
        try:
            device_id = int(source.split(":")[1])
        except (IndexError, ValueError):
            device_id = 0
        return WebcamSource(device_id=device_id)
    
    if source.startswith(("rtsp://", "http://", "https://")):
        return VideoStreamSource(stream_url=source)
    
    # Assume it's a file path
    return VideoFileSource(file_path=source, loop=loop)
