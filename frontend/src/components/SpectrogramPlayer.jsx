import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause } from 'lucide-react';
import clsx from 'clsx';

export default function SpectrogramPlayer({ spectrogramUrl, audioUrl }) {
    const audioRef = useRef(null);
    const containerRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [progress, setProgress] = useState(0); // 0 to 1

    const togglePlay = () => {
        if (!audioRef.current) return;
        if (isPlaying) {
            audioRef.current.pause();
        } else {
            audioRef.current.play();
        }
    };

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const onPlay = () => setIsPlaying(true);
        const onPause = () => setIsPlaying(false);
        const onEnded = () => { setIsPlaying(false); setProgress(0); };

        // High fidelity time tracking loop
        let rafId;
        const updateLoop = () => {
            if (audio.duration) {
                setProgress(audio.currentTime / audio.duration);
            }
            rafId = requestAnimationFrame(updateLoop);
        };
        updateLoop();

        audio.addEventListener('play', onPlay);
        audio.addEventListener('pause', onPause);
        audio.addEventListener('ended', onEnded);

        return () => {
            cancelAnimationFrame(rafId);
            audio.removeEventListener('play', onPlay);
            audio.removeEventListener('pause', onPause);
            audio.removeEventListener('ended', onEnded);
        };
    }, [audioUrl]);

    // Handle audio/spectrogram url changes - reset
    useEffect(() => {
        if (audioRef.current) {
            audioRef.current.load();
            setIsPlaying(false);
            setProgress(0);
        }
    }, [audioUrl]);

    // Seek handler
    const handleSeek = (e) => {
        if (!audioRef.current || !containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const p = Math.max(0, Math.min(1, x / rect.width));

        if (audioRef.current.duration) {
            audioRef.current.currentTime = p * audioRef.current.duration;
            setProgress(p);
        }
    };

    return (
        <div className="relative w-full h-full group rounded-2xl overflow-hidden bg-black border border-gray-800 shadow-2xl flex flex-col">
            {/* Spectrogram Container */}
            <div
                ref={containerRef}
                className="relative w-full h-full bg-gray-900 cursor-crosshair overflow-hidden flex-1"
                onClick={handleSeek}
            >
                {spectrogramUrl ? (
                    <img
                        src={spectrogramUrl}
                        className="w-full h-full object-fill select-none"
                        alt="Spectrogram"
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-700 font-mono text-sm">
                        Awaiting Signal...
                    </div>
                )}

                {/* Scanning Line */}
                <div
                    className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_15px_rgba(255,255,255,0.8)] pointer-events-none transition-transform duration-75 will-change-transform"
                    style={{ left: `${progress * 100}%` }}
                />

                {/* Play Overlay (if paused) */}
                {!isPlaying && spectrogramUrl && (
                    <div className="absolute inset-0 bg-black/20 flex items-center justify-center pointer-events-none">
                        <div className="w-16 h-16 rounded-full bg-white/10 backdrop-blur-sm flex items-center justify-center border border-white/20">
                            <Play className="w-6 h-6 text-white ml-1" />
                        </div>
                    </div>
                )}
            </div>

            {/* Audio Element (Hidden) */}
            <audio ref={audioRef} src={audioUrl} preload="auto" />

            {/* Custom Controls Bar */}
            <div className="absolute bottom-0 inset-x-0 h-16 bg-gradient-to-t from-black/90 to-transparent flex items-end p-4">
                <button
                    onClick={togglePlay}
                    className="text-white hover:text-purple-400 transition-colors"
                >
                    {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
                </button>
                {/* Simple waveform reference or just time */}
                <div className="ml-4 text-xs font-mono text-gray-400 mb-1">
                    {audioRef.current ? (progress * audioRef.current.duration).toFixed(1) : "0.0"}s
                </div>
            </div>
        </div>
    );
}
