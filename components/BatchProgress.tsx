import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  CheckCircle2,
  XCircle,
  Loader2,
  Download,
  AlertCircle,
  Clock,
  Zap,
  FileAudio,
  Timer,
  Activity,
} from 'lucide-react';
import { api } from '../services/api';
import { BatchFileInfo } from '../types';

interface Props {
  files: File[];
  engine?: string;
  whisperModel?: string;
  onDone: () => void;
  onError: (msg: string) => void;
}

type UploadState = 'uploading' | 'processing' | 'done';

interface FileTracker {
  file: File;
  projectId?: string;
  uploadError?: string;
}

const CONCURRENT_UPLOADS = 3;

// Helper: format seconds to MM:SS or HH:MM:SS
function formatTime(seconds: number): string {
  if (seconds < 0) seconds = 0;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

// Helper: format file size
function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} КБ`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} ГБ`;
}

// Status to step number (for mini progress bar per file)
// Pipeline: convert → transcribe → done (S3 upload removed — gRPC streams directly)
const STEP_ORDER: Record<string, number> = {
  queued: 0,
  downloading: 1,
  converting: 1,
  transcribing: 2,
  completed: 3,
  error: -1,
};

const TOTAL_STEPS = 3;

const STEP_LABELS: Record<string, string> = {
  queued: 'В очереди',
  downloading: 'Скачивание...',
  converting: 'Конвертация аудио...',
  transcribing: 'Распознавание речи...',
  completed: 'Готово',
  error: 'Ошибка',
};

export const BatchProgress: React.FC<Props> = ({ files, engine = 'whisper', whisperModel = 'small', onDone, onError }) => {
  const [state, setState] = useState<UploadState>('uploading');
  const [trackers, setTrackers] = useState<FileTracker[]>(() =>
    files.map(f => ({ file: f }))
  );
  const [uploadedCount, setUploadedCount] = useState(0);
  const [currentUploadName, setCurrentUploadName] = useState('');
  const [batchFiles, setBatchFiles] = useState<BatchFileInfo[]>([]);
  const [completedCount, setCompletedCount] = useState(0);
  const [errorCount, setErrorCount] = useState(0);
  const [isDownloading, setIsDownloading] = useState(false);
  const [startTime] = useState(() => Date.now());
  const [elapsedSec, setElapsedSec] = useState(0);
  const [logMessages, setLogMessages] = useState<string[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval>>();
  const timerRef = useRef<ReturnType<typeof setInterval>>();
  const logEndRef = useRef<HTMLDivElement>(null);
  const prevBatchFilesRef = useRef<BatchFileInfo[]>([]);

  const addLog = useCallback((msg: string) => {
    const time = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setLogMessages(prev => [...prev.slice(-200), `[${time}] ${msg}`]);
  }, []);

  // Timer: update elapsed every second
  useEffect(() => {
    timerRef.current = setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [startTime]);

  // Auto-scroll log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logMessages]);

  // Phase 1: Preload Whisper model (if needed), then upload files to server
  useEffect(() => {
    let cancelled = false;

    const uploadAll = async () => {
      const queue = [...files];
      const updated = [...trackers];
      let doneCount = 0;

      // Pre-download Whisper model before starting uploads
      if (engine === 'whisper') {
        addLog(`Проверка модели Whisper '${whisperModel}'...`);
        try {
          await api.preloadWhisperModel(whisperModel);
          addLog(`✓ Модель Whisper '${whisperModel}' готова`);
        } catch (e: any) {
          addLog(`✗ Ошибка загрузки модели Whisper: ${e.message}`);
          onError(`Не удалось загрузить модель Whisper '${whisperModel}': ${e.message}`);
          setState('done');
          return;
        }
        if (cancelled) return;
      }

      addLog(`Начинаем загрузку ${queue.length} файлов на сервер...`);

      const uploadOne = async (index: number) => {
        if (cancelled) return;
        const file = queue[index];
        setCurrentUploadName(file.name);
        addLog(`Загрузка: ${file.name} (${formatSize(file.size)})`);
        try {
          const projectId = await api.batchUploadFile(file, engine, whisperModel);
          updated[index] = { ...updated[index], projectId };
          addLog(`✓ Загружен: ${file.name}`);
        } catch (e: any) {
          updated[index] = { ...updated[index], uploadError: e.message };
          addLog(`✗ Ошибка загрузки: ${file.name} — ${e.message}`);
        }
        doneCount++;
        if (!cancelled) {
          setTrackers([...updated]);
          setUploadedCount(doneCount);
        }
      };

      // Upload in batches of CONCURRENT_UPLOADS
      for (let i = 0; i < queue.length; i += CONCURRENT_UPLOADS) {
        if (cancelled) break;
        const batch = [];
        for (let j = i; j < Math.min(i + CONCURRENT_UPLOADS, queue.length); j++) {
          batch.push(uploadOne(j));
        }
        await Promise.all(batch);
      }

      if (!cancelled) {
        setTrackers([...updated]);
        setCurrentUploadName('');
        addLog(`Все файлы загружены на сервер. Начинается обработка...`);
        setState('processing');
      }
    };

    uploadAll();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Phase 2: Poll for processing status
  useEffect(() => {
    if (state !== 'processing') return;

    const projectIds = trackers
      .filter(t => t.projectId)
      .map(t => t.projectId!);

    if (projectIds.length === 0) {
      setState('done');
      return;
    }

    const poll = async () => {
      try {
        const status = await api.batchStatus(projectIds);
        setBatchFiles(prev => {
          // Detect status changes and log them
          const prevMap = new Map(prevBatchFilesRef.current.map(f => [f.id, f]));
          for (const file of status.files) {
            const prevFile = prevMap.get(file.id);
            if (prevFile && prevFile.status !== file.status) {
              if (file.status === 'completed') {
                addLog(`✓ Обработан: ${file.filename}`);
              } else if (file.status === 'error') {
                addLog(`✗ Ошибка: ${file.filename} — ${file.error || 'неизвестная ошибка'}`);
              } else {
                const label = STEP_LABELS[file.status] || file.status_label;
                addLog(`${file.filename} → ${label}`);
              }
            }
          }
          prevBatchFilesRef.current = status.files;
          return status.files;
        });
        setCompletedCount(status.completed);
        setErrorCount(status.errors);

        if (status.in_progress === 0) {
          addLog(`Обработка завершена! Готово: ${status.completed}, ошибок: ${status.errors}`);
          setState('done');
          if (pollRef.current) clearInterval(pollRef.current);
          if (timerRef.current) clearInterval(timerRef.current);
        }
      } catch {
        // Ignore polling errors, retry next interval
      }
    };

    poll();
    pollRef.current = setInterval(poll, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [state, trackers, addLog]);

  const handleDownload = useCallback(async () => {
    setIsDownloading(true);
    try {
      const projectIds = trackers
        .filter(t => t.projectId)
        .map(t => t.projectId!);

      const blob = await api.batchDownload(projectIds);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'transcripts.zip';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (e: any) {
      onError(e?.message || 'Ошибка скачивания архива');
    } finally {
      setIsDownloading(false);
    }
  }, [trackers, onError]);

  const totalFiles = files.length;
  const uploadPercent = Math.round((uploadedCount / totalFiles) * 100);
  const doneTotal = completedCount + errorCount;
  const processPercent = totalFiles > 0 ? Math.round((doneTotal / totalFiles) * 100) : 0;

  // ETA calculation
  const estimatedRemainingSec = (() => {
    if (state === 'uploading') {
      if (uploadedCount === 0) return null;
      const perFile = elapsedSec / uploadedCount;
      return Math.round(perFile * (totalFiles - uploadedCount));
    }
    if (state === 'processing') {
      if (doneTotal === 0) return null;
      const perFile = elapsedSec / doneTotal;
      return Math.round(perFile * (totalFiles - doneTotal));
    }
    return null;
  })();

  // Speed
  const filesPerMin = elapsedSec > 10
    ? ((state === 'uploading' ? uploadedCount : doneTotal) / (elapsedSec / 60)).toFixed(1)
    : null;

  // Currently active files (being processed right now)
  const activeFiles = batchFiles.filter(
    f => f.status !== 'completed' && f.status !== 'error' && f.status !== 'queued'
  );

  // Build unified file status list
  const fileStatuses = trackers.map(t => {
    if (t.uploadError) {
      return { name: t.file.name, status: 'error', label: 'Ошибка загрузки', error: t.uploadError, step: -1 };
    }
    if (!t.projectId) {
      if (state === 'uploading') {
        return { name: t.file.name, status: 'waiting', label: 'Ожидание...', step: 0 };
      }
      return { name: t.file.name, status: 'error', label: 'Не загружен', step: -1 };
    }
    const batchFile = batchFiles.find(bf => bf.id === t.projectId);
    if (batchFile) {
      const step = STEP_ORDER[batchFile.status] ?? 0;
      const label = STEP_LABELS[batchFile.status] || batchFile.status_label;
      return { name: t.file.name, status: batchFile.status, label, error: batchFile.error, step };
    }
    if (state === 'uploading') {
      return { name: t.file.name, status: 'uploaded', label: 'Загружен на сервер', step: 0 };
    }
    return { name: t.file.name, status: 'queued', label: 'В очереди', step: 0 };
  });

  return (
    <div className="flex flex-col h-full p-4 lg:p-6 gap-4">
      {/* === TOP: Main progress card === */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
        {/* Header row */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            {state !== 'done' ? (
              <div className="relative">
                <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                  <Activity className="w-5 h-5 text-blue-600" />
                </div>
                <span className="absolute -top-0.5 -right-0.5 flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-blue-500"></span>
                </span>
              </div>
            ) : (
              <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                <CheckCircle2 className="w-5 h-5 text-green-600" />
              </div>
            )}
            <div>
              <h2 className="text-lg font-bold text-gray-900">
                {state === 'uploading' && 'Загрузка файлов на сервер'}
                {state === 'processing' && 'Обработка файлов'}
                {state === 'done' && 'Обработка завершена!'}
              </h2>
              <p className="text-sm text-gray-500">
                {state === 'uploading' && `${uploadedCount} из ${totalFiles} загружено`}
                {state === 'processing' && `${completedCount} из ${totalFiles} обработано`}
                {state === 'done' && `${completedCount} успешно, ${errorCount} с ошибками`}
              </p>
            </div>
          </div>

          {state === 'done' && (
            <button
              onClick={handleDownload}
              disabled={isDownloading || completedCount === 0}
              className="flex items-center gap-2 px-5 py-2.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white rounded-lg font-medium transition-colors"
            >
              <Download className="w-4 h-4" />
              {isDownloading ? 'Скачивание...' : `Скачать ZIP (${completedCount})`}
            </button>
          )}
        </div>

        {/* Main progress bar */}
        <div className="mb-3">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>
              {state === 'uploading'
                ? `Загрузка: ${uploadedCount} / ${totalFiles}`
                : `Готово: ${completedCount} / ${totalFiles}`}
              {errorCount > 0 && <span className="text-red-500 ml-1">(ошибок: {errorCount})</span>}
            </span>
            <span className="font-mono font-semibold text-gray-700">
              {state === 'uploading' ? uploadPercent : processPercent}%
            </span>
          </div>
          <div className="w-full h-3 bg-gray-100 rounded-full overflow-hidden">
            {state === 'done' ? (
              <div className="h-full bg-green-500 rounded-full transition-all duration-500" style={{ width: '100%' }} />
            ) : (
              <div className="h-full rounded-full transition-all duration-500 relative overflow-hidden" style={{
                width: `${state === 'uploading' ? uploadPercent : processPercent}%`,
                background: 'linear-gradient(90deg, #3b82f6, #6366f1)',
              }}>
                {/* Animated shimmer */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/25 to-transparent animate-[shimmer_2s_infinite]" style={{
                  animation: 'shimmer 2s infinite',
                }} />
              </div>
            )}
          </div>
        </div>

        {/* Stats row */}
        <div className="flex flex-wrap gap-4 text-xs">
          <div className="flex items-center gap-1.5 text-gray-600">
            <Clock className="w-3.5 h-3.5" />
            <span>Прошло: <strong className="font-mono">{formatTime(elapsedSec)}</strong></span>
          </div>
          {estimatedRemainingSec !== null && (
            <div className="flex items-center gap-1.5 text-gray-600">
              <Timer className="w-3.5 h-3.5" />
              <span>Осталось: <strong className="font-mono">~{formatTime(estimatedRemainingSec)}</strong></span>
            </div>
          )}
          {filesPerMin && (
            <div className="flex items-center gap-1.5 text-gray-600">
              <Zap className="w-3.5 h-3.5" />
              <span>Скорость: <strong>{filesPerMin} файлов/мин</strong></span>
            </div>
          )}
          {state !== 'done' && (
            <div className="flex items-center gap-1.5 text-blue-600">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span>Обновление каждые 2 сек</span>
            </div>
          )}
        </div>
      </div>

      {/* === MIDDLE: Currently active + currently uploading === */}
      {state === 'uploading' && currentUploadName && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 flex items-center gap-3">
          <Loader2 className="w-4 h-4 text-amber-600 animate-spin flex-shrink-0" />
          <span className="text-sm text-amber-800 truncate">
            Загружается: <strong>{currentUploadName}</strong>
          </span>
        </div>
      )}

      {state === 'processing' && activeFiles.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-3">
          <div className="text-xs font-medium text-blue-700 mb-2 flex items-center gap-1.5">
            <FileAudio className="w-3.5 h-3.5" />
            Сейчас обрабатываются:
          </div>
          <div className="space-y-1.5">
            {activeFiles.map(af => (
              <div key={af.id} className="flex items-center gap-2">
                <Loader2 className="w-3 h-3 text-blue-500 animate-spin flex-shrink-0" />
                <span className="text-sm text-blue-900 truncate flex-1">{af.filename}</span>
                <span className="text-xs text-blue-600 bg-blue-100 px-2 py-0.5 rounded-full flex-shrink-0">
                  {STEP_LABELS[af.status] || af.status_label}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* === BOTTOM: Two columns — file list + log === */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0">
        {/* File list */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden flex flex-col min-h-0">
          <div className="px-4 py-2.5 border-b border-gray-200 bg-gray-50 flex items-center justify-between flex-shrink-0">
            <span className="text-sm font-medium text-gray-700">
              Файлы ({totalFiles})
            </span>
            <div className="flex gap-3 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3 text-green-500" />
                {completedCount}
              </span>
              {errorCount > 0 && (
                <span className="flex items-center gap-1">
                  <XCircle className="w-3 h-3 text-red-500" />
                  {errorCount}
                </span>
              )}
            </div>
          </div>
          <div className="flex-1 overflow-y-auto divide-y divide-gray-50">
            {fileStatuses.map((fs, i) => (
              <div key={i} className={`flex items-center px-4 py-2 ${
                fs.status === 'completed' ? 'bg-green-50/50' :
                fs.status === 'error' ? 'bg-red-50/50' : ''
              }`}>
                <div className="flex-shrink-0 mr-3">
                  {fs.status === 'completed' ? <CheckCircle2 className="w-4 h-4 text-green-500" /> :
                   fs.status === 'error' ? <XCircle className="w-4 h-4 text-red-500" /> :
                   fs.status === 'waiting' || fs.status === 'queued' ? (
                     <div className="w-4 h-4 rounded-full border-2 border-gray-200" />
                   ) : (
                     <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                   )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-800 truncate">{fs.name}</p>
                  {/* Mini step indicator for active files */}
                  {fs.step > 0 && fs.step < TOTAL_STEPS && (
                    <div className="flex gap-0.5 mt-1">
                      {Array.from({ length: TOTAL_STEPS }, (_, i) => i + 1).map(s => (
                        <div key={s} className={`h-1 flex-1 rounded-full ${
                          s <= fs.step
                            ? 'bg-blue-500'
                            : s === fs.step + 1
                            ? 'bg-blue-200 animate-pulse'
                            : 'bg-gray-200'
                        }`} />
                      ))}
                    </div>
                  )}
                </div>
                <div className="ml-3 flex-shrink-0">
                  <span className={`text-xs ${
                    fs.status === 'completed' ? 'text-green-600 font-medium' :
                    fs.status === 'error' ? 'text-red-600 font-medium' :
                    fs.step > 0 ? 'text-blue-600' :
                    'text-gray-400'
                  }`}>
                    {fs.label}
                  </span>
                </div>
                {fs.error && (
                  <div className="ml-2 flex-shrink-0" title={fs.error}>
                    <AlertCircle className="w-3.5 h-3.5 text-red-400" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Activity log */}
        <div className="bg-gray-900 rounded-xl shadow-sm border border-gray-700 overflow-hidden flex flex-col min-h-0">
          <div className="px-4 py-2.5 border-b border-gray-700 bg-gray-800 flex items-center gap-2 flex-shrink-0">
            <Activity className="w-3.5 h-3.5 text-green-400" />
            <span className="text-sm font-medium text-gray-300">Журнал</span>
            {state !== 'done' && (
              <span className="ml-auto flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
            )}
          </div>
          <div className="flex-1 overflow-y-auto p-3 font-mono text-xs leading-relaxed">
            {logMessages.length === 0 ? (
              <p className="text-gray-600">Ожидание...</p>
            ) : (
              logMessages.map((msg, i) => (
                <div key={i} className={`${
                  msg.includes('✓') ? 'text-green-400' :
                  msg.includes('✗') ? 'text-red-400' :
                  msg.includes('→') ? 'text-blue-400' :
                  'text-gray-400'
                }`}>
                  {msg}
                </div>
              ))
            )}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>

      {/* Footer */}
      {state === 'done' && (
        <div className="flex items-center justify-between">
          <button
            onClick={onDone}
            className="text-sm text-gray-500 hover:text-gray-700"
          >
            Вернуться на главную
          </button>
          <span className="text-xs text-gray-400">
            Общее время: {formatTime(elapsedSec)}
          </span>
        </div>
      )}

      {/* CSS for shimmer animation */}
      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
      `}</style>
    </div>
  );
};
