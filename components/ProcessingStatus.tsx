import React from 'react';
import { Loader2, XCircle, Check } from 'lucide-react';

interface Props {
  currentStep: string;
  progressPercent: number;
  onCancel: () => void;
}

const PIPELINE_STEPS = [
  { key: 'downloading', label: 'Скачивание' },
  { key: 'converting', label: 'Конвертация' },
  { key: 'transcribing', label: 'Распознавание' },
  { key: 'formatting', label: 'Формирование' },
];

const STEP_MAP: Record<string, number> = {
  'Скачивание...': 0,
  'Скачивание файла...': 0,
  'Конвертация аудио...': 1,
  'Конвертация...': 1,
  'Распознавание речи...': 2,
  'Распознавание...': 2,
  'Формирование результата...': 3,
  'Формирование...': 3,
};

export const ProcessingStatus: React.FC<Props> = ({ currentStep, progressPercent, onCancel }) => {
  const activeIdx = STEP_MAP[currentStep] ?? -1;

  return (
    <div className="flex flex-col items-center justify-center h-full space-y-8 px-4">
      <div className="text-center">
        <h3 className="text-xl font-semibold text-gray-800">Обработка медиа</h3>
        {progressPercent > 0 && (
          <p className="text-2xl font-bold text-blue-600 mt-1">{progressPercent}%</p>
        )}
      </div>

      {/* Vertical pipeline */}
      <div className="space-y-0">
        {PIPELINE_STEPS.map((step, idx) => {
          const state = activeIdx > idx ? 'done' : activeIdx === idx ? 'active' : 'pending';
          return (
            <div key={step.key} className="flex items-start gap-3">
              <div className="flex flex-col items-center">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${
                  state === 'done'
                    ? 'bg-green-100 text-green-600'
                    : state === 'active'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-400'
                }`}>
                  {state === 'done' ? (
                    <Check className="w-4 h-4" />
                  ) : state === 'active' ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <span className="text-xs font-semibold">{idx + 1}</span>
                  )}
                </div>
                {idx < PIPELINE_STEPS.length - 1 && (
                  <div className={`w-0.5 h-6 ${
                    state === 'done' ? 'bg-green-400' : 'bg-gray-200'
                  }`} />
                )}
              </div>
              <div className="pt-1.5">
                <span className={`text-sm font-medium ${
                  state === 'active' ? 'text-gray-900' :
                  state === 'done' ? 'text-gray-600' :
                  'text-gray-400'
                }`}>
                  {step.label}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="w-64 h-1.5 bg-gray-200 rounded-full overflow-hidden">
        {progressPercent > 0 ? (
          <div
            className="h-full bg-blue-600 rounded-full transition-all duration-300"
            style={{ width: `${progressPercent}%` }}
          />
        ) : (
          <div className="h-full bg-blue-600 rounded-full animate-progress-indeterminate" />
        )}
      </div>

      {currentStep && activeIdx === -1 && (
        <p className="text-sm text-gray-500 font-mono">{currentStep}</p>
      )}

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
