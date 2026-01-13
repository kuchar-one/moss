import React from 'react';
import { ArrowLeft, ArrowRight, Zap } from 'lucide-react';

export default function OptimizationController({ onSteer, onPareto }) {
    // onSteer(direction): -1 (Audio) or +1 (Image)
    // Actually we want weights?
    // Let's implement "Better Sound" vs "Better Image"

    return (
        <div className="flex flex-col items-center space-y-6 w-full max-w-2xl mx-auto">
            <div className="flex items-center justify-between w-full gap-4">

                <button
                    onClick={() => onSteer('audio')}
                    className="flex-1 py-4 px-6 rounded-xl bg-emerald-900/30 border border-emerald-800/50 hover:bg-emerald-900/50 hover:border-emerald-500 transition-all group"
                >
                    <div className="flex flex-col items-center space-y-2">
                        <span className="text-emerald-400 font-medium">Better Sound</span>
                        <span className="text-xs text-emerald-600/60 uppercase tracking-widest group-hover:text-emerald-500">Reduce Distortion</span>
                    </div>
                </button>

                <div className="text-gray-600 font-mono text-xs">VS</div>

                <button
                    onClick={() => onSteer('image')}
                    className="flex-1 py-4 px-6 rounded-xl bg-purple-900/30 border border-purple-800/50 hover:bg-purple-900/50 hover:border-purple-500 transition-all group"
                >
                    <div className="flex flex-col items-center space-y-2">
                        <span className="text-purple-400 font-medium">Better Image</span>
                        <span className="text-xs text-purple-600/60 uppercase tracking-widest group-hover:text-purple-500">Enhance Visuals</span>
                    </div>
                </button>
            </div>

            <div className="w-full pt-4 border-t border-gray-800/50">
                <button
                    onClick={onPareto}
                    className="w-full py-3 rounded-lg bg-gray-900 border border-gray-700 text-gray-400 hover:text-white hover:border-white transition-colors flex items-center justify-center space-x-2"
                >
                    <Zap className="w-4 h-4" />
                    <span>Map Full Pareto Frontier</span>
                </button>
            </div>
        </div>
    );
}
