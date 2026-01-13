import React, { useEffect, useState } from 'react';
import { getResources, getFileURL } from '../api';
import clsx from 'clsx';
import { Check, Music, Image as ImageIcon } from 'lucide-react';

export default function SelectionGrid({ onSelect, onStart }) {
    const [data, setData] = useState({ images: [], audio: [] });
    const [selectedImage, setSelectedImage] = useState(null);
    const [selectedAudio, setSelectedAudio] = useState(null);

    useEffect(() => {
        getResources().then(setData);
    }, []);

    const canStart = selectedImage && selectedAudio;

    return (
        <div className="flex flex-col h-full max-w-6xl mx-auto p-8 space-y-12">
            <div className="text-center space-y-2">
                <h1 className="text-4xl font-light tracking-tight text-white">Start New Optimization</h1>
                <p className="text-gray-400">Select a target image and audio source to begin.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-16">
                {/* Images */}
                <div className="space-y-6">
                    <h2 className="flex items-center text-xl font-medium text-purple-300">
                        <ImageIcon className="mr-3 w-6 h-6" /> Target Image
                    </h2>
                    <div className="grid grid-cols-2 gap-4">
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
                                <div className="absolute bottom-0 inset-x-0 bg-black/60 p-2 text-xs truncate">
                                    {img}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Audio */}
                <div className="space-y-6">
                    <h2 className="flex items-center text-xl font-medium text-emerald-300">
                        <Music className="mr-3 w-6 h-6" /> Target Audio
                    </h2>
                    <div className="space-y-3">
                        {data.audio.map(aud => (
                            <div
                                key={aud}
                                onClick={() => setSelectedAudio(aud)}
                                className={clsx(
                                    "relative flex items-center p-4 rounded-lg cursor-pointer transition-all border",
                                    selectedAudio === aud
                                        ? "bg-emerald-950/30 border-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.2)]"
                                        : "bg-gray-800/40 border-gray-700 hover:bg-gray-800"
                                )}
                            >
                                <div className="flex-1 min-w-0">
                                    <div className="font-medium text-gray-200 truncate">{aud}</div>
                                </div>
                                {selectedAudio === aud && (
                                    <Check className="w-5 h-5 text-emerald-400 ml-3" />
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="flex justify-center pt-8">
                <button
                    disabled={!canStart}
                    onClick={() => onStart(selectedImage, selectedAudio)}
                    className={clsx(
                        "px-10 py-4 rounded-full text-lg font-semibold tracking-wide transition-all",
                        canStart
                            ? "bg-white text-black hover:scale-105 shadow-[0_0_30px_rgba(255,255,255,0.3)]"
                            : "bg-gray-800 text-gray-500 cursor-not-allowed"
                    )}
                >
                    Ignite Optimization
                </button>
            </div>
        </div>
    );
}
