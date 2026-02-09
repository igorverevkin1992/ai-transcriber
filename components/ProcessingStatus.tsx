import React from 'react';
import { Loader2, XCircle } from 'lucide-react';

interface Props {
  currentStep: string;
  progressPercent: number;
  onCancel: () => void;
}

export const ProcessingStatus: React.FC<Props> = ({ currentStep, progressPercent, onCancel }) => {
  const hasPercent = progressPercent > 0;

  return (
    <div className="flex flex-col items-center justify-center h-full space-y-6 px-4">
      <div className="relative w-24 h-24">
        <div className="absolute inset-0 border-4 border-gray-200 rounded-full"></div>
        <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          {hasPercent ? (
            <span className="text-lg font-bold text-blue-600">{progressPercent}%</span>
          ) : (
            <Loader2 className="w-8 h-8 text-blue-600 animate-pulse" />
          )}
        </div>
      </div>
      <div className="text-center">
        <h3 className="text-xl font-semibold text-gray-800">Обработка медиа</h3>
        <p className="mt-2 text-gray-500 font-mono text-sm">{currentStep || 'Подготовка...'}</p>
      </div>
      <div className="w-64 h-1.5 bg-gray-200 rounded-full overflow-hidden">
        {hasPercent ? (
          <div
            className="h-full bg-blue-600 rounded-full transition-all duration-300"
            style={{ width: `${progressPercent}%` }}
          ></div>
        ) : (
          <div className="h-full bg-blue-600 rounded-full animate-progress-indeterminate"></div>
        )}
      </div>
      <button
        onClick={onCancel}
        className="flex items-center gap-2 px-4 py-2 text-sm text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-md transition-colors"
      >
        <XCircle className="w-4 h-4" />
        Отменить
      </button>
    </div>
  );
};
