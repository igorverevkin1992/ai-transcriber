import React, { useState } from 'react';
import { UploadCloud, Link as LinkIcon, AlertCircle } from 'lucide-react';

interface Props {
  onUpload: (link: string) => void;
}

export const UploadForm: React.FC<Props> = ({ onUpload }) => {
  const [link, setLink] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
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
    onUpload(trimmed);
  };

  return (
    <div className="flex flex-col items-center justify-center h-full max-w-2xl mx-auto px-4">
      <div className="bg-white p-8 rounded-2xl shadow-xl w-full border border-gray-100">
        <div className="text-center mb-8">
          <div className="bg-blue-50 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
            <UploadCloud className="w-8 h-8 text-blue-600" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">ABTGS</h1>
          <p className="text-gray-500 mt-2">Automated Broadcast Transcript Generation System</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Ссылка на исходник (Yandex.Disk)
            </label>
            <div className="relative rounded-md shadow-sm">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <LinkIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
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
            className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
          >
            Обработать материал
          </button>
        </form>

        <div className="mt-6 border-t border-gray-100 pt-6">
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Технические требования</h4>
          <ul className="text-xs text-gray-500 space-y-1 list-disc pl-4">
            <li>Поддерживаемые форматы: .mp3, .wav, .mov, .mxf, .mp4, .wmv, .avi, .mkv, .ogg, .flac</li>
            <li>Максимальный размер файла: 1 ГБ</li>
            <li>Расчет таймкода: SMPTE 25 FPS (PAL)</li>
          </ul>
        </div>
      </div>
    </div>
  );
};