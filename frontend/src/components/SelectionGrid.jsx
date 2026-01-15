import React, { useEffect, useState, useRef } from 'react';
import { getResources, getFileURL } from '../api';
import clsx from 'clsx';
import { Check, Music, Image as ImageIcon, Play, Square } from 'lucide-react';

export default function SelectionGrid({ onSelect, onStart }) {
    const [data, setData] = useState({ images: [], audio: [] });
    const [selectedImage, setSelectedImage] = useState(null);
    const [selectedAudio, setSelectedAudio] = useState(null);
    const [playingAudio, setPlayingAudio] = useState(null);
    const audioRef = useRef(null);

    const playAudio = (audioFile, e) => {
        e.stopPropagation(); // Don't trigger selection

        // Stop currently playing audio if any
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current = null;
        }

        if (playingAudio === audioFile) {
            // Was playing this file, now stopped
            setPlayingAudio(null);
            return;
        }

        // Play new audio
        const audio = new Audio(getFileURL(audioFile));
        audioRef.current = audio;
        setPlayingAudio(audioFile);

        audio.addEventListener('ended', () => {
            setPlayingAudio(null);
            audioRef.current = null;
        });

        audio.play().catch(err => {
            console.error('Failed to play audio:', err);
            setPlayingAudio(null);
            audioRef.current = null;
        });
    };

    // Cleanup audio on unmount
    useEffect(() => {
        return () => {
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        getResources().then(setData);
    }, []);

    const canStart = selectedImage && selectedAudio;

    return (
        <div className="flex flex-col h-full max-w-5xl mx-auto p-6">
            <div className="text-center space-y-2 mb-8 flex-shrink-0">
                <h1 className="text-3xl font-bold tracking-[0.1em] text-transparent bg-clip-text bg-gradient-to-r from-white via-purple-100 to-white/60">
                    Multi-Objective Sound Synthesis
                </h1>
                <p className="text-white/40 font-light text-sm">Select a target image and audio source to begin.</p>
            </div>

            <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-2 gap-8 overflow-hidden">
                <div className="flex flex-col min-h-0">
                    <h2 className="flex items-center text-sm font-medium text-white/60 mb-3 flex-shrink-0">
                        <div className="w-2 h-2 rounded-full bg-purple-500 mr-2 shadow-[0_0_8px_rgba(168,85,247,0.5)]" />
                        Target Image
                    </h2>
                    <div className="flex-1 overflow-y-auto min-h-0 pr-2">
                        <div className="grid grid-cols-2 gap-3">
                            {data.images.map(img => (
                                <div
                                    key={img}
                                    onClick={() => setSelectedImage(img)}
                                    className={clsx(
                                        "group relative aspect-square rounded-xl overflow-hidden cursor-pointer transition-all border-2",
                                        selectedImage === img ? "border-purple-500 shadow-[0_0_20px_rgba(168,85,247,0.4)]" : "border-transparent hover:border-gray-600"
                                    )}
                                >
                                    <img
                                        src={getFileURL(img)}
                                        alt={img}
                                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                                    />
                                    {selectedImage === img && (
                                        <div className="absolute inset-0 bg-purple-900/40 flex items-center justify-center">
                                            <Check className="w-10 h-10 text-white drop-shadow-lg" />
                                        </div>
                                    )}
                                    <div className="absolute bottom-0 inset-x-0 bg-black/70 backdrop-blur-sm p-2 text-xs text-white/70 truncate font-mono">
                                        {img.split('/').pop()}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="flex flex-col min-h-0">
                    <h2 className="flex items-center text-sm font-medium text-white/60 mb-3 flex-shrink-0">
                        <div className="w-2 h-2 rounded-full bg-purple-500 mr-2 shadow-[0_0_8px_rgba(168,85,247,0.5)]" />
                        Target Audio
                    </h2>
                    <div className="flex-1 overflow-y-auto min-h-0 pr-2 space-y-2">
                        {data.audio.map(aud => (
                            <div
                                key={aud}
                                onClick={() => setSelectedAudio(aud)}
                                className={clsx(
                                    "relative flex items-center p-3 rounded-lg cursor-pointer transition-all border",
                                    selectedAudio === aud
                                        ? "bg-purple-950/30 border-purple-500/50 shadow-[0_0_15px_rgba(168,85,247,0.2)]"
                                        : "bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20"
                                )}
                            >
                                <button
                                    onClick={(e) => playAudio(aud, e)}
                                    className={clsx(
                                        "w-8 h-8 rounded-full flex items-center justify-center mr-3 transition-all duration-200 flex-shrink-0",
                                        playingAudio === aud
                                            ? "bg-purple-500 text-white shadow-[0_0_12px_rgba(168,85,247,0.6)]"
                                            : "bg-white/10 text-white/60 hover:bg-purple-500/30 hover:text-purple-300"
                                    )}
                                    aria-label={playingAudio === aud ? "Stop audio" : "Play audio preview"}
                                >
                                    {playingAudio === aud ? (
                                        <Square className="w-3 h-3" />
                                    ) : (
                                        <Play className="w-3 h-3 ml-0.5" />
                                    )}
                                </button>
                                <div className="flex-1 min-w-0">
                                    <div className="font-medium text-white/80 truncate font-mono text-sm">{aud.split('/').pop()}</div>
                                </div>
                                {selectedAudio === aud && (
                                    <Check className="w-5 h-5 text-purple-400 ml-3" />
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="flex-shrink-0 flex justify-center gap-4 py-4 border-t border-white/5 mt-4">
                <button
                    disabled={!canStart}
                    onClick={() => onStart(selectedImage, selectedAudio, 'single')}
                    className={clsx(
                        "px-8 py-4 rounded-full text-lg font-semibold tracking-wide transition-all duration-300",
                        canStart
                            ? "bg-white/10 text-white hover:bg-white/20 hover:scale-105 border border-white/10"
                            : "bg-white/5 text-white/30 cursor-not-allowed"
                    )}
                >
                    Quick Start
                </button>
                <button
                    disabled={!canStart}
                    onClick={() => onStart(selectedImage, selectedAudio, 'pareto')}
                    className={clsx(
                        "px-10 py-4 rounded-full text-lg font-semibold tracking-wide transition-all duration-300",
                        canStart
                            ? "bg-gradient-to-r from-purple-600 to-pink-500 text-white hover:scale-105 shadow-[0_0_30px_rgba(168,85,247,0.4)] hover:shadow-[0_0_40px_rgba(236,72,153,0.6)]"
                            : "bg-white/5 text-white/30 cursor-not-allowed"
                    )}
                >
                    Map Full Frontier
                </button>
            </div>
        </div>
    );
}
