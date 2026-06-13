import React, { useState, useRef } from 'react';
import { UploadCloud, Link as LinkIcon, AlertCircle, X, Cpu, Cloud, ChevronDown, ChevronUp } from 'lucide-react';

export type EngineType = 'whisper' | 'speechkit';
export type WhisperModel = 'small' | 'medium' | 'large';

interface Props {
  onStartBatch: (files: File[], engine: EngineType, whisperModel: WhisperModel) => void;
  onUploadLink: (link: string, engine: EngineType, whisperModel: WhisperModel) => void;
}

const ALLOWED_EXTENSIONS = new Set([
  '.mp3', '.wav', '.mov', '.mxf', '.mp4', '.wmv', '.avi', '.mkv', '.ogg', '.flac',
]);

const MAX_FILE_SIZE = 1024 * 1024 * 1024; // 1 GB

const WHISPER_MODELS: { value: WhisperModel; label: string; desc: string }[] = [
  { value: 'small', label: 'Small', desc: '~1 ГБ RAM, ~10 мин/час аудио, хорошее качество' },
  { value: 'medium', label: 'Medium', desc: '~2 ГБ RAM, ~30 мин/час аудио, высокое качество' },
  { value: 'large', label: 'Large', desc: '~4 ГБ RAM, ~60 мин/час аудио, лучшее качество' },
];

export const BatchUploadForm: React.FC<Props> = ({ onStartBatch, onUploadLink }) => {
  const [tab, setTab] = useState<'files' | 'link'>('files');
  const [files, setFiles] = useState<File[]>([]);
  const [link, setLink] = useState('');
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [engine, setEngine] = useState<EngineType>('whisper');
  const [whisperModel, setWhisperModel] = useState<WhisperModel>('small');
  const [showSettings, setShowSettings] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);

  const filterValidFiles = (fileList: FileList | File[]): { valid: File[]; oversized: string[] } => {
    const arr = Array.from(fileList);
    const valid: File[] = [];
    const oversized: string[] = [];
    for (const f of arr) {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase();
      if (!ALLOWED_EXTENSIONS.has(ext)) continue;
      if (f.size > MAX_FILE_SIZE) {
        oversized.push(f.name);
      } else {
        valid.push(f);
      }
    }
    return { valid, oversized };
  };

  const addFiles = (newFiles: File[]) => {
    const { valid, oversized } = filterValidFiles(newFiles);
    if (oversized.length > 0) {
      setError(`Файлы превышают лимит 1 ГБ: ${oversized.join(', ')}`);
    }
    if (valid.length === 0 && oversized.length === 0) {
      setError('Не найдено поддерживаемых медиафайлов');
      return;
    }
    if (valid.length > 0) {
      setFiles(prev => {
        const existing = new Set(prev.map(f => `${f.name}_${f.size}`));
        const unique = valid.filter(f => !existing.has(`${f.name}_${f.size}`));
        return [...prev, ...unique];
      });
      if (oversized.length === 0) setError('');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;
    addFiles(Array.from(e.target.files));
    e.target.value = '';
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    if (e.dataTransfer.items?.length) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    dragCounter.current = 0;
    if (e.dataTransfer.files?.length) {
      addFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleRemoveFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmitFiles = () => {
    if (files.length === 0) {
      setError('Выберите файлы для обработки');
      return;
    }
    onStartBatch(files, engine, whisperModel);
  };

  const handleSubmitLink = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = link.trim();
    if (!trimmed) {
      setError('Вставьте ссылку на файл');
      return;
    }
    if (!trimmed.includes('yadi.sk') && !trimmed.includes('disk.yandex')) {
      setError('Пожалуйста, введите корректную публичную ссылку на Яндекс.Диск');
      return;
    }
    onUploadLink(trimmed, engine, whisperModel);
  };

  const totalSize = files.reduce((acc, f) => acc + f.size, 0);
  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} КБ`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} ГБ`;
  };

  const engineSummary = engine === 'whisper' ? `Whisper (${whisperModel})` : 'SpeechKit';

  // Настройки движка — общие для обеих вкладок (файлы и ссылка).
  const engineSettings = (
    <div className="mb-5">
      <button
        onClick={() => setShowSettings(!showSettings)}
        className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors w-full"
      >
        {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        <span>Движок распознавания: <strong>{engineSummary}</strong></span>
      </button>

      {showSettings && (
        <div className="mt-3 p-4 border border-gray-200 rounded-lg bg-gray-50 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => setEngine('whisper')}
              className={`flex items-center gap-3 p-3 rounded-lg border-2 text-left transition-colors ${
                engine === 'whisper' ? 'border-green-500 bg-green-50' : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <Cpu className={`w-5 h-5 flex-shrink-0 ${engine === 'whisper' ? 'text-green-600' : 'text-gray-400'}`} />
              <div>
                <div className={`text-sm font-semibold ${engine === 'whisper' ? 'text-green-900' : 'text-gray-700'}`}>Whisper</div>
                <div className="text-xs text-gray-500">Бесплатно, локально</div>
              </div>
            </button>
            <button
              type="button"
              onClick={() => setEngine('speechkit')}
              className={`flex items-center gap-3 p-3 rounded-lg border-2 text-left transition-colors ${
                engine === 'speechkit' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <Cloud className={`w-5 h-5 flex-shrink-0 ${engine === 'speechkit' ? 'text-blue-600' : 'text-gray-400'}`} />
              <div>
                <div className={`text-sm font-semibold ${engine === 'speechkit' ? 'text-blue-900' : 'text-gray-700'}`}>SpeechKit</div>
                <div className="text-xs text-gray-500">Облако, диаризация</div>
              </div>
            </button>
          </div>

          {engine === 'whisper' && (
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1.5">Модель Whisper</label>
              <select
                value={whisperModel}
                onChange={e => setWhisperModel(e.target.value as WhisperModel)}
                aria-label="Выбор модели Whisper"
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm bg-white focus:ring-2 focus:ring-green-500 focus:border-green-500"
              >
                {WHISPER_MODELS.map(m => (
                  <option key={m.value} value={m.value}>{m.label} — {m.desc}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}
    </div>
  );

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="flex-1 flex items-start justify-center py-6 px-4">
        <div className="bg-white p-6 md:p-8 rounded-2xl shadow-xl w-full max-w-3xl border border-gray-100">
          <div className="text-center mb-6">
            <div className="bg-blue-50 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <UploadCloud className="w-8 h-8 text-blue-600" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900">Загрузка материалов</h1>
            <p className="text-gray-500 mt-2">Автоматическая генерация монтажных листов</p>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-gray-200 mb-6">
            <button
              onClick={() => { setTab('files'); setError(''); }}
              className={`flex-1 pb-3 text-sm font-medium text-center border-b-2 transition-colors ${
                tab === 'files'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Загрузить файлы
            </button>
            <button
              onClick={() => { setTab('link'); setError(''); }}
              className={`flex-1 pb-3 text-sm font-medium text-center border-b-2 transition-colors ${
                tab === 'link'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Ссылка на Яндекс.Диск
            </button>
          </div>

          {/* Общие настройки движка (применяются к обеим вкладкам) */}
          {engineSettings}

          {/* Tab: Files */}
          {tab === 'files' && (
            <>
              <input
                ref={inputRef}
                type="file"
                multiple
                accept=".mp3,.wav,.mov,.mxf,.mp4,.wmv,.avi,.mkv,.ogg,.flac"
                className="hidden"
                aria-label="Выберите медиафайлы"
                onChange={handleFileSelect}
              />

              <div
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                className="relative"
              >
                {files.length === 0 ? (
                  <button
                    onClick={() => inputRef.current?.click()}
                    className={`w-full border-2 border-dashed rounded-xl p-10 transition-colors flex flex-col items-center gap-3 ${
                      isDragging
                        ? 'border-blue-500 bg-blue-50 scale-[1.02]'
                        : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
                    }`}
                  >
                    <UploadCloud className={`w-10 h-10 ${isDragging ? 'text-blue-500' : 'text-gray-400'}`} />
                    <span className={`font-medium ${isDragging ? 'text-blue-700' : 'text-gray-600'}`}>
                      {isDragging ? 'Отпустите файлы здесь' : 'Перетащите файлы сюда или нажмите для выбора'}
                    </span>
                    <span className="text-xs text-gray-400">
                      Поддержка: .mp3, .wav, .mov, .mxf, .mp4 и др. (до 1 ГБ на файл)
                    </span>
                  </button>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between bg-blue-50 rounded-lg px-4 py-3">
                      <div>
                        <span className="font-semibold text-blue-900">{files.length} файлов</span>
                        <span className="text-blue-600 ml-2 text-sm">({formatSize(totalSize)})</span>
                      </div>
                      <button
                        onClick={() => inputRef.current?.click()}
                        className="text-sm text-blue-600 hover:text-blue-800 font-medium"
                      >
                        Добавить ещё
                      </button>
                    </div>

                    <div className="max-h-64 overflow-y-auto border border-gray-200 rounded-lg divide-y divide-gray-100">
                      {files.map((file, i) => (
                        <div key={`${file.name}_${file.size}_${file.lastModified}`} className="flex items-center justify-between px-3 py-2 hover:bg-gray-50">
                          <div className="flex-1 min-w-0">
                            <p className="text-sm text-gray-700 truncate">{file.name}</p>
                            <p className="text-xs text-gray-400">{formatSize(file.size)}</p>
                          </div>
                          <button
                            onClick={() => handleRemoveFile(i)}
                            aria-label={`Удалить ${file.name}`}
                            className="ml-2 p-1 text-gray-400 hover:text-red-500"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      ))}
                    </div>

                    {isDragging && (
                      <div className="absolute inset-0 bg-blue-50/80 border-2 border-dashed border-blue-500 rounded-xl flex items-center justify-center z-10">
                        <span className="text-blue-600 font-medium">Отпустите чтобы добавить файлы</span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {error && (
                <div className="mt-3 text-sm text-red-600 flex items-center gap-1">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  {error}
                </div>
              )}

              <button
                onClick={handleSubmitFiles}
                disabled={files.length === 0}
                className="w-full mt-6 flex justify-center py-3 px-4 rounded-lg text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                Обработать {files.length > 0 ? `${files.length} файлов` : 'файлы'}
              </button>
            </>
          )}

          {/* Tab: Link */}
          {tab === 'link' && (
            <form onSubmit={handleSubmitLink} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Публичная ссылка на Яндекс.Диск
                </label>
                <div className="relative rounded-md shadow-sm">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <LinkIcon className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    type="text"
                    aria-label="Ссылка на Яндекс.Диск"
                    className="focus:ring-blue-500 focus:border-blue-500 block w-full pl-10 sm:text-sm border-gray-300 rounded-md py-3 border"
                    placeholder="https://yadi.sk/d/..."
                    value={link}
                    onChange={(e) => {
                      setLink(e.target.value);
                      setError('');
                    }}
                  />
                </div>
                {error && (
                  <div className="mt-2 text-sm text-red-600 flex items-center gap-1">
                    <AlertCircle className="w-4 h-4" />
                    {error}
                  </div>
                )}
              </div>

              <button
                type="submit"
                className="w-full flex justify-center py-3 px-4 rounded-lg text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 transition-colors"
              >
                Обработать материал
              </button>

              <div className="border-t border-gray-100 pt-4">
                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Технические требования</h4>
                <ul className="text-xs text-gray-500 space-y-1 list-disc pl-4">
                  <li>Поддерживаемые форматы: .mp3, .wav, .mov, .mxf, .mp4, .wmv, .avi, .mkv, .ogg, .flac</li>
                  <li>Максимальный размер файла: 1 ГБ</li>
                  <li>Расчет таймкода: SMPTE 25 FPS (PAL)</li>
                </ul>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};
