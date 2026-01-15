import axios from 'axios';

const API_Base = '/api';

const client = axios.create({
    baseURL: API_Base,
});

export const getResources = async () => {
    const response = await client.get('/resources');
    return response.data;
};

export const startOptimization = async (params) => {
    // params: { image_path, audio_path, mode, weights, seed_task_id, seed_index }
    const response = await client.post('/optimize', params);
    return response.data;
};

export const getStatus = async (taskId) => {
    const response = await client.get(`/status/${taskId}`);
    return response.data;
};

export const getFileURL = (path) => {
    // path usually "output/UUID/spectrogram.png" or "data/image.jpg"
    // API serves:
    // /files/data/image.jpg
    // /files/output/UUID/spectrogram.png

    // We need to handle this mapping.
    // The backend assumes "data_dir" for source files.

    if (path.startsWith('output') || path.includes('/output/')) {
        // If we receive "output/UUID/..." -> map to /files/output/UUID/...
        // But backend task["result_path"] is absolute path. 
        // Wait, backend `result_path` returned from `/status` is ALREADY relative `/files/output/...`
        // See backend/main.py: `result_path = f"/files/output/{task_id}"`
        // So if we have result_path, use it directly.
        return `${API_Base}${path}`;
    }

    // For input files (resources endpoint returns filenames like "cat.jpg")
    // We map them to /files/data/cat.jpg
    return `${API_Base}/files/data/${path}`;
};
