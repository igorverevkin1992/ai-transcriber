import React, { useState, useRef } from 'react';
import { UploadCloud, FolderOpen, AlertCircle, X } from 'lucide-react';

interface Props {
  onStartBatch: (files: File[]) => void;
  onSwitchToSingle: () => void;
}

const ALLOWED_EXTENSIONS = new Set([
  '.mp3', '.wav', '.mov', '.mxf', '.mp4', '.wmv', '.avi', '.mkv', '.ogg', '.flac',
]);

export const BatchUploadForm: React.FC<Props> = ({ onStartBatch, onSwitchToSingle }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const filterValidFiles = (fileList: FileList | File[]): File[] => {
    const arr = Array.from(fileList);
    return arr.filter(f => {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase();
      return ALLOWED_EXTENSIONS.has(ext);
    });
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;
    const valid = filterValidFiles(e.target.files);
    if (valid.length === 0) {
      setError('Не найдено поддерживаемых медиафайлов');
      return;
    }
    setFiles(valid);
    setError('');
  };

  const handleRemoveFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    if (files.length === 0) {
      setError('Выберите файлы для обработки');
      return;
    }
    onStartBatch(files);
  };

  const totalSize = files.reduce((acc, f) => acc + f.size, 0);
  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} КБ`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} ГБ`;
  };

  return (
    <div className="flex flex-col items-center justify-center h-full max-w-3xl mx-auto px-4">
      <div className="bg-white p-8 rounded-2xl shadow-xl w-full border border-gray-100">
        <div className="text-center mb-6">
          <div className="bg-purple-50 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
            <FolderOpen className="w-8 h-8 text-purple-600" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">Пакетная обработка</h1>
          <p className="text-gray-500 mt-2">Загрузите несколько файлов для автоматической расшифровки</p>
        </div>

        {/* Hidden file input */}
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".mp3,.wav,.mov,.mxf,.mp4,.wmv,.avi,.mkv,.ogg,.flac"
          className="hidden"
          onChange={handleFileSelect}
        />

        {files.length === 0 ? (
          <button
            onClick={() => inputRef.current?.click()}
            className="w-full border-2 border-dashed border-gray-300 rounded-xl p-10 hover:border-purple-400 hover:bg-purple-50 transition-colors flex flex-col items-center gap-3"
          >
            <UploadCloud className="w-10 h-10 text-gray-400" />
            <span className="text-gray-600 font-medium">Нажмите чтобы выбрать файлы</span>
            <span className="text-xs text-gray-400">Поддержка: .mp3, .wav, .mov, .mxf, .mp4, .wmv, .avi, .mkv, .ogg, .flac</span>
          </button>
        ) : (
          <div className="space-y-4">
            {/* Summary */}
            <div className="flex items-center justify-between bg-purple-50 rounded-lg px-4 py-3">
              <div>
                <span className="font-semibold text-purple-900">{files.length} файлов</span>
                <span className="text-purple-600 ml-2 text-sm">({formatSize(totalSize)})</span>
              </div>
              <button
                onClick={() => inputRef.current?.click()}
                className="text-sm text-purple-600 hover:text-purple-800 font-medium"
              >
                Изменить
              </button>
            </div>

            {/* File list (scrollable) */}
            <div className="max-h-64 overflow-y-auto border border-gray-200 rounded-lg divide-y divide-gray-100">
              {files.map((file, i) => (
                <div key={i} className="flex items-center justify-between px-3 py-2 hover:bg-gray-50">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-700 truncate">{file.name}</p>
                    <p className="text-xs text-gray-400">{formatSize(file.size)}</p>
                  </div>
                  <button
                    onClick={() => handleRemoveFile(i)}
                    className="ml-2 p-1 text-gray-400 hover:text-red-500"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="mt-3 text-sm text-red-600 flex items-center gap-1">
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={files.length === 0}
          className="w-full mt-6 flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          Начать обработку ({files.length} файлов)
        </button>

        <div className="mt-4 text-center">
          <button
            onClick={onSwitchToSingle}
            className="text-sm text-gray-500 hover:text-gray-700"
          >
            Обработать один файл по ссылке
          </button>
        </div>
      </div>
    </div>
  );
};
