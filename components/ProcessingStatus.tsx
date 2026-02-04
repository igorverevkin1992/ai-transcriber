import React from 'react';
import { Loader2 } from 'lucide-react';

interface Props {
  currentStep: string;
}

export const ProcessingStatus: React.FC<Props> = ({ currentStep }) => {
  return (
    <div className="flex flex-col items-center justify-center h-full space-y-6">
      <div className="relative w-24 h-24">
        <div className="absolute inset-0 border-4 border-gray-200 rounded-full"></div>
        <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <Loader2 className="w-8 h-8 text-blue-600 animate-pulse" />
        </div>
      </div>
      <div className="text-center">
        <h3 className="text-xl font-semibold text-gray-800">Обработка медиа</h3>
        <p className="mt-2 text-gray-500 font-mono text-sm">{currentStep}</p>
      </div>
      <div className="w-64 h-1 bg-gray-200 rounded-full overflow-hidden">
        <div className="h-full bg-blue-600 rounded-full animate-progress-indeterminate"></div>
      </div>
    </div>
  );
};