import React, { useEffect, useState, useRef } from 'react';
import { getStatus, startOptimization, getFileURL } from '../api';
import SpectrogramPlayer from './SpectrogramPlayer';
import OptimizationController from './OptimizationController';
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Loader2 } from 'lucide-react';
import clsx from 'clsx';

export default function OptimizerView({ initialTaskId, imagePath, audioPath, onBack }) {
    const [currentTaskId, setCurrentTaskId] = useState(initialTaskId);
    const [history, setHistory] = useState([]); // [{taskId, F: [vis, aud], status, result_path}]
    const [taskState, setTaskState] = useState({ status: 'pending', progress: 0 });
    const [currentWeights, setCurrentWeights] = useState([0.5, 0.5]); // [Vis, Aud]
    const pollingRef = useRef(null);

    // Poll current task
    useEffect(() => {
        if (!currentTaskId) return;

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
                        // Ensure we strictly have numbers
                        const cleanMetrics = metrics.map(n => Number(n) || 0);

                        return [...prev, {
                            ...data,
                            result_metrics: cleanMetrics,
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
        setCurrentTaskId(res.task_id);
    };

    const handlePareto = async () => {
        const params = {
            image_path: imagePath,
            audio_path: audioPath,
            mode: 'pareto',
            seed_task_id: currentTaskId
        };
        const res = await startOptimization(params);
        setCurrentTaskId(res.task_id);
    };

    // Calculate URL for current result
    // result_path from backend is relative "/files/output/..."
    const resultUrl = taskState.result_path ? getFileURL(taskState.result_path) : null;
    const specUrl = resultUrl ? `${resultUrl}/spectrogram.png` : null;
    const wavUrl = resultUrl ? `${resultUrl}/output.wav` : null;

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
                <div className="lg:col-span-2 flex flex-col space-y-6">
                    {/* Resizable Container */}
                    <div className="relative w-full resize overflow-hidden min-h-[300px] h-auto bg-black rounded-3xl border border-white/10 shadow-2xl group/resize">                         <div className="w-full h-full relative"> {/* Inner Wrapper */}
                        {taskState.status === 'completed' && specUrl ? (
                            <SpectrogramPlayer
                                spectrogramUrl={specUrl}
                                audioUrl={wavUrl}
                            />
                        ) : (
                            <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4 min-h-[300px]">
                                <Loader2 className="w-12 h-12 text-purple-500 animate-spin" />
                                <div className="text-purple-300 font-light animate-pulse">
                                    Optimizing Spectrum... {Math.round(taskState.progress * 100)}%
                                </div>
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
                </div>

                {/* Sidebar: History & Plot */}
                <div className="flex flex-col space-y-6">
                    <div className="bg-gray-900/50 rounded-2xl p-6 border border-white/5 h-64">
                        <h3 className="text-white/60 text-sm font-medium mb-4">Evolution History</h3>
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
                                <XAxis type="number" dataKey="x" name="Vis Loss" label={{ value: 'Visual Loss', position: 'insideBottom', offset: -20, fill: '#666' }} domain={['auto', 'auto']} tick={{ fill: '#666', fontSize: 12 }} />
                                <YAxis type="number" dataKey="y" name="Aud Loss" label={{ value: 'Audio Loss', angle: -90, position: 'insideLeft', fill: '#666' }} domain={['auto', 'auto']} tick={{ fill: '#666' }} />
                                <Tooltip
                                    cursor={{ strokeDasharray: '3 3' }}
                                    contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }}
                                />
                                <Scatter name="History" data={history.map((h, i) => {
                                    const [vis, aud] = h.result_metrics || [0, 0];
                                    // Ensure safe numbers
                                    return {
                                        x: Number(vis) || 0,
                                        y: Number(aud) || 0,
                                        id: h.taskId
                                    };
                                })} fill="#8884d8">
                                    {history.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.taskId === currentTaskId ? '#ffffff' : '#a855f7'} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>

                    </div>

                    <div className="flex-1 bg-gray-900/50 rounded-2xl p-6 border border-white/5 overflow-y-auto">
                        <h3 className="text-white/60 text-sm font-medium mb-4">Generations</h3>
                        <div className="space-y-3">
                            {history.map((h, i) => (
                                <div
                                    key={h.taskId}
                                    onClick={() => setCurrentTaskId(h.taskId)}
                                    className={clsx(
                                        "p-3 rounded-lg cursor-pointer transition-all border",
                                        h.taskId === currentTaskId
                                            ? "bg-purple-900/20 border-purple-500/50"
                                            : "bg-black/20 border-transparent hover:border-white/10"
                                    )}
                                >
                                    <div className="text-xs text-gray-400 font-mono">Generation {i + 1}</div>
                                    <div className="text-sm text-white truncate">
                                        {h.mode === 'single' ? 'Single Point' : 'Pareto Front'}
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
