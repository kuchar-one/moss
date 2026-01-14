import React, { useEffect, useState, useRef } from 'react';
import { getStatus, startOptimization, getFileURL } from '../api';
import SpectrogramPlayer from './SpectrogramPlayer';
import OptimizationController from './OptimizationController';
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Loader2 } from 'lucide-react';
import clsx from 'clsx';

export default function OptimizerView({ initialTaskId, imagePath, audioPath, onBack }) {
    // Selection state now tracks both Task ID and Index (for Pareto)
    const [selection, setSelection] = useState({ taskId: initialTaskId, index: 0 });
    const [history, setHistory] = useState([]); // [{taskId, F: [vis, aud], status, result_path}]
    const [taskState, setTaskState] = useState({ status: 'pending', progress: 0 });
    const [currentWeights, setCurrentWeights] = useState([0.5, 0.5]); // [Vis, Aud]
    const pollingRef = useRef(null);

    const currentTaskId = selection.taskId;
    const currentIndex = selection.index || 0;

    // Poll current task
    useEffect(() => {
        if (!currentTaskId) return;

        // Reset state when switching tasks
        setTaskState({ status: 'pending', progress: 0 });

        const poll = async () => {
            try {
                const data = await getStatus(currentTaskId);
                setTaskState(data);

                if (data.status === 'completed') {
                    // Stop polling immediately
                    clearInterval(pollingRef.current);

                    // Mark as done in history
                    setHistory(prev => {
                        // Prevent duplicates
                        if (prev.find(h => h.taskId === currentTaskId)) return prev;

                        const metrics = data.result_metrics || [0, 0];
                        // Handle formatting safely for both 1D and 2D arrays
                        let cleanMetrics;
                        if (Array.isArray(metrics[0])) {
                            // 2D Array (Pareto)
                            cleanMetrics = metrics.map(point => {
                                if (Array.isArray(point)) return point.map(n => Number(n) || 0);
                                return [0, 0];
                            });
                        } else {
                            // 1D Array (Single)
                            cleanMetrics = metrics.map(n => Number(n) || 0);
                        }

                        return [...prev, {
                            ...data,
                            taskId: currentTaskId, // Explicitly save taskId
                            result_metrics: cleanMetrics,
                            weights: data.mode === 'single' ? currentWeights : null, // Store weights for history display
                            mode: data.mode, // Ensure mode is saved
                            imagePath,
                            audioPath
                        }];
                    });
                } else if (data.status === 'failed') {
                    clearInterval(pollingRef.current);
                }
            } catch (e) {
                console.error("Polling error", e);
            }
        };

        pollingRef.current = setInterval(poll, 1000);
        poll(); // immediate

        return () => clearInterval(pollingRef.current);
    }, [currentTaskId]);

    const handleSteer = async (direction) => {
        // Incremental Steering
        let [w_vis, w_aud] = currentWeights;
        const step = 0.1;

        if (direction === 'audio') {
            // Better Audio -> Higher Audio Weight (Minimize Audio Loss)
            w_aud += step;
            w_vis -= step;
        } else {
            // Better Image -> Higher Visual Weight
            w_vis += step;
            w_aud -= step;
        }

        // Clamp
        w_vis = Math.max(0.05, Math.min(0.95, w_vis));
        w_aud = 1.0 - w_vis; // Ensure sum is 1.0

        const newWeights = [w_vis, w_aud];
        setCurrentWeights(newWeights);

        console.log(`Steering ${direction}:`, newWeights);

        // Start new optimization
        // Seed with current result?
        const params = {
            image_path: imagePath,
            audio_path: audioPath,
            mode: 'single',
            weights: newWeights,
            seed_task_id: currentTaskId,
            seed_index: 0 // Single point result is always idx 0
        };

        const res = await startOptimization(params);
        setSelection({ taskId: res.task_id, index: 0 });
    };

    const handlePareto = async () => {
        const params = {
            image_path: imagePath,
            audio_path: audioPath,
            mode: 'pareto',
            seed_task_id: currentTaskId
        };
        const res = await startOptimization(params);
        setSelection({ taskId: res.task_id, index: 0 });
    };

    // Calculate URL for current result
    // If Pareto -> Use streaming endpoints
    // If Single -> Use static files

    // Check current task mode from history cache or taskState
    const currentTaskFromHistory = history.find(h => h.taskId === currentTaskId);
    const isPareto = currentTaskFromHistory?.mode === 'pareto' || taskState.mode === 'pareto';

    const getDynamicURL = () => {
        // Use backend port 8000 directly or via proxy? API is usually relative in dev but we need full URL if on different port?
        // api.js getFileURL uses localhost:8000.
        const baseUrl = getFileURL('').replace('/files/output/', ''); // Hacky way to get base API url
        return baseUrl;
    }

    // We can rely on relative paths if setup correctly, but let's be safe.
    // getFileURL returns "http://localhost:8000/files/output/..."
    // We want "http://localhost:8000/results/..."

    const apiBase = "http://localhost:8000"; // Hardcoded for now matching default_api



    const [mediaLoaded, setMediaLoaded] = useState(false);

    useEffect(() => {
        if (taskState.status !== 'completed') {
            setMediaLoaded(false);
            return;
        }

        let sUrl = null;
        let wUrl = null;

        if (isPareto) {
            sUrl = `${apiBase}/results/${currentTaskId}/${currentIndex}/spectrogram`;
            wUrl = `${apiBase}/results/${currentTaskId}/${currentIndex}/audio`;
        } else {
            const resultUrl = taskState.result_path ? getFileURL(taskState.result_path) : null;
            sUrl = resultUrl ? `${resultUrl}/spectrogram.png` : null;
            wUrl = resultUrl ? `${resultUrl}/output.wav` : null;
        }

        setMediaLoaded(false);

        // Preload
        if (sUrl && wUrl) {
            Promise.all([
                new Promise((resolve, reject) => {
                    const img = new Image();
                    img.onload = resolve;
                    img.onerror = reject;
                    img.src = sUrl;
                }),
                new Promise((resolve, reject) => {
                    const aud = new Audio();
                    aud.onloadedmetadata = resolve; // or oncanplaythrough
                    aud.onerror = (e) => {
                        console.warn("Audio load error, proceeding anyway", e);
                        resolve(); // Resolve anyway to show at least image
                    };
                    aud.src = wUrl;
                })
            ]).then(() => {
                setMediaLoaded(true);
            }).catch(e => {
                console.error("Media load failed", e);
                // Even on fail, maybe show what we have?
                setMediaLoaded(true);
            });
        }
    }, [taskState.status, currentTaskId, currentIndex, isPareto, taskState.result_path]);



    // Calculate for Render
    let specUrl = null;
    let wavUrl = null;
    if (taskState.status === 'completed') {
        if (isPareto) {
            specUrl = `${apiBase}/results/${currentTaskId}/${currentIndex}/spectrogram`;
            wavUrl = `${apiBase}/results/${currentTaskId}/${currentIndex}/audio`;
        } else {
            const resultUrl = taskState.result_path ? getFileURL(taskState.result_path) : null;
            specUrl = resultUrl ? `${resultUrl}/spectrogram.png` : null;
            wavUrl = resultUrl ? `${resultUrl}/output.wav` : null;
        }
    }

    return (
        <div className="flex flex-col h-full max-w-7xl mx-auto p-6 space-y-8">
            <header className="flex items-center justify-between text-white/50 border-b border-white/10 pb-4">
                <button onClick={onBack} className="hover:text-white transition-colors">
                    &larr; Start Over
                </button>
                <div className="font-mono text-xs">
                    TASK: {currentTaskId.split('-')[0]} • STATUS: {taskState.status.toUpperCase()}
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-full">
                {/* Main Visualizer Area */}
                <div className="lg:col-span-2 flex flex-col space-y-6 relative z-30">
                    {/* Resizable Container */}
                    <div className="relative w-full resize overflow-hidden min-h-[300px] h-auto bg-black rounded-3xl border border-white/10 shadow-2xl group/resize">                         <div className="w-full h-full relative"> {/* Inner Wrapper */}
                        {taskState.status === 'completed' && specUrl ? (
                            mediaLoaded ? (
                                <SpectrogramPlayer
                                    spectrogramUrl={specUrl}
                                    audioUrl={wavUrl}
                                />
                            ) : (
                                <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4 min-h-[300px] bg-black/80 backdrop-blur-sm z-50">
                                    <Loader2 className="w-10 h-10 text-white/50 animate-spin" />
                                    <div className="text-white/50 text-sm font-mono tracking-widest">LOADING MEDIA...</div>
                                </div>
                            )
                        ) : (
                            <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4 min-h-[300px]">
                                <Loader2 className={`w-12 h-12 animate-spin ${isPareto ? 'text-pink-500' : 'text-purple-500'}`} />
                                <div className={`font-light animate-pulse ${isPareto ? 'text-pink-300' : 'text-purple-300'}`}>
                                    {isPareto
                                        ? `Mapping Pareto Frontier... ${Math.round(taskState.progress * 100)}%`
                                        : `Optimizing... ${Math.round(taskState.progress * 100)}%`
                                    }
                                </div>
                                {isPareto && taskState.progress < 0.5 && (
                                    <div className="text-white/30 text-xs font-mono">Phase 1: Seeding</div>
                                )}
                                {isPareto && taskState.progress >= 0.5 && (
                                    <div className="text-white/30 text-xs font-mono">Phase 2: Evolution</div>
                                )}
                            </div>
                        )}
                    </div>
                        {/* Resize Handle Hint */}
                        <div className="absolute bottom-1 right-1 w-4 h-4 border-r-2 border-b-2 border-gray-500/50 pointer-events-none group-hover/resize:border-white/80 transition-colors" />
                    </div>

                    {/* Controls */}
                    <div className="bg-black/40 backdrop-blur-md rounded-2xl p-6 border border-white/5">
                        <div className="text-xs text-center text-gray-500 mb-2 font-mono">
                            Current Balance: Vis {(currentWeights[0] * 100).toFixed(0)}% / Aud {(currentWeights[1] * 100).toFixed(0)}%
                        </div>
                        <OptimizationController onSteer={handleSteer} onPareto={handlePareto} />
                    </div>

                    {/* Evolution History Animation - Moved to Left Column */}
                    {isPareto && taskState.status === 'completed' && (
                        <div className="bg-black/40 backdrop-blur-md rounded-2xl p-6 border border-white/5">
                            <h3 className="text-white/60 text-sm font-medium mb-4">Evolution Replay</h3>
                            <div className="rounded-lg overflow-hidden border border-white/10 bg-black">
                                <video
                                    key={currentTaskId} // Force reload on new task
                                    controls
                                    className="w-full h-auto"
                                    src={`http://localhost:8000/files/output/${currentTaskId}/history.mp4`}
                                >
                                    Your browser does not support the video tag.
                                </video>
                            </div>
                        </div>
                    )}
                </div>

                {/* Sidebar: History & Plot - Added z-index and relative positioning to prevent overlap issues */}
                <div className="flex flex-col space-y-6 relative z-0 pointer-events-none lg:pointer-events-auto">
                    <div className="bg-gray-900/50 rounded-2xl p-6 border border-white/5 h-64 pointer-events-auto flex flex-col">
                        <h3 className="text-white/60 text-sm font-medium mb-4 flex-shrink-0">Evolution History</h3>
                        <div className="flex-1 min-h-0 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 30, left: 10 }}>
                                    <XAxis type="number" dataKey="x" name="Vis Loss" label={{ value: 'Visual Loss', position: 'insideBottom', offset: -20, fill: '#666' }} domain={['auto', 'auto']} tick={{ fill: '#666', fontSize: 12 }} />
                                    <YAxis type="number" dataKey="y" name="Aud Loss" label={{ value: 'Audio Loss', angle: -90, position: 'insideBottomLeft', fill: '#666' }} domain={['auto', 'auto']} tick={{ fill: '#666' }} />
                                    <Tooltip
                                        cursor={{ strokeDasharray: '3 3' }}
                                        contentStyle={{ backgroundColor: '#09090b', border: '1px solid #27272a', color: '#e4e4e7', borderRadius: '8px', fontSize: '12px' }}
                                        itemStyle={{ color: '#e4e4e7' }}
                                        labelStyle={{ color: '#a1a1aa', marginBottom: '4px' }}
                                    />
                                    <Scatter
                                        name="History"
                                        onClick={(node) => {
                                            // Handle click to select individual
                                            // node contains { x, y, id, index, ... }
                                            if (node && node.id) {
                                                setSelection({ taskId: node.id, index: node.index });
                                            }
                                        }}
                                        data={history.flatMap((h) => {
                                            const metrics = h.result_metrics;

                                            // Handle missing data
                                            if (!metrics) return [];

                                            if (Array.isArray(metrics[0])) {
                                                return metrics.map(([vis, aud], idx) => ({
                                                    x: Number(vis) || 0,
                                                    y: Number(aud) || 0,
                                                    id: h.taskId, // All share same ID
                                                    index: idx, // Pass index for selection
                                                    isPareto: true
                                                }));
                                            }

                                            // Handle Single Point [v, a]
                                            const [vis, aud] = metrics;
                                            return [{
                                                x: Number(vis) || 0,
                                                y: Number(aud) || 0,
                                                id: h.taskId,
                                                index: 0,
                                                isPareto: false
                                            }];
                                        })} fill="#8884d8" style={{ cursor: 'pointer' }}>
                                        {history.flatMap(h => {
                                            // We need to match the flatMap structure above for cells
                                            const metrics = h.result_metrics || [];
                                            const isParetoMode = Array.isArray(metrics[0]);
                                            const counts = isParetoMode ? metrics.length : 1;
                                            return Array(counts).fill(h).map((_, idx) => ({ h, idx }));
                                        }).map((entry, index) => {
                                            const isSelected = entry.h.taskId === currentTaskId && (entry.h.mode === 'single' || entry.idx === currentIndex);
                                            return (
                                                <Cell
                                                    key={`cell-${index}`}
                                                    fill={isSelected ? '#ffffff' : entry.h.mode === 'pareto' ? '#4ade80' : '#a855f7'}
                                                    opacity={entry.h.mode === 'pareto' ? 0.6 : 1.0}
                                                />
                                            )
                                        })}
                                    </Scatter>
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="flex-1 bg-gray-900/50 rounded-2xl p-6 border border-white/5 overflow-y-auto min-h-[200px] max-h-[60vh]">
                        <h3 className="text-white/60 text-sm font-medium mb-4">Individuals</h3>
                        <div className="space-y-3">
                            {history.flatMap((h, histIdx) => {
                                // Unpack Pareto Items
                                const metrics = h.result_metrics;
                                if (h.mode === 'pareto' && Array.isArray(metrics) && Array.isArray(metrics[0])) {
                                    // Map first, then sort
                                    const items = metrics.map((m, idx) => ({
                                        taskId: h.taskId,
                                        index: idx,
                                        mode: 'pareto',
                                        metrics: m,
                                        label: `Individual ${idx + 1}`
                                    }));
                                    // Sort by Visual Loss (metrics[0]) Ascending
                                    return items.sort((a, b) => a.metrics[0] - b.metrics[0]);
                                }
                                return [{
                                    taskId: h.taskId,
                                    index: 0,
                                    mode: 'single',
                                    metrics: metrics,
                                    weights: h.weights,
                                    label: `Generation ${histIdx + 1}`
                                }];
                            }).map((item, i) => (
                                <div
                                    key={`${item.taskId}-${item.index}`}
                                    onClick={() => setSelection({ taskId: item.taskId, index: item.index })}
                                    className={clsx(
                                        "p-3 rounded-lg cursor-pointer transition-all border",
                                        (item.taskId === currentTaskId && item.index === currentIndex)
                                            ? "bg-purple-900/20 border-purple-500/50"
                                            : "bg-black/20 border-transparent hover:border-white/10"
                                    )}
                                >
                                    <div className="text-xs text-gray-400 font-mono">{item.label}</div>
                                    <div className="text-sm text-white truncate font-mono">
                                        {item.mode === 'single'
                                            ? `Vis: ${(item.weights ? item.weights[0] * 100 : item.metrics[0] * 100).toFixed(0)}% / Aud: ${(item.weights ? item.weights[1] * 100 : item.metrics[1] * 100).toFixed(0)}%`
                                            : <span className="text-xs text-green-400">
                                                V: {Number(item.metrics[0]).toFixed(3)} | A: {Number(item.metrics[1]).toFixed(3)}
                                            </span>
                                        }
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>


                </div>
            </div>
        </div>
    );
}
