import React, { useEffect } from 'react';
import { X, AlertCircle, CheckCircle2 } from 'lucide-react';

export interface ToastMessage {
  id: string;
  type: 'error' | 'success';
  text: string;
}

interface Props {
  messages: ToastMessage[];
  onDismiss: (id: string) => void;
}

export const ToastContainer: React.FC<Props> = ({ messages, onDismiss }) => {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {messages.map(msg => (
        <ToastItem key={msg.id} message={msg} onDismiss={onDismiss} />
      ))}
    </div>
  );
};

const ToastItem: React.FC<{ message: ToastMessage; onDismiss: (id: string) => void }> = ({
  message,
  onDismiss,
}) => {
  useEffect(() => {
    const timer = setTimeout(() => onDismiss(message.id), 6000);
    return () => clearTimeout(timer);
  }, [message.id, onDismiss]);

  const isError = message.type === 'error';

  return (
    <div
      className={`flex items-start gap-3 p-4 rounded-lg shadow-lg border animate-fade-in-up ${
        isError ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'
      }`}
    >
      {isError ? (
        <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
      ) : (
        <CheckCircle2 className="w-5 h-5 text-green-500 shrink-0 mt-0.5" />
      )}
      <p className={`text-sm flex-1 ${isError ? 'text-red-800' : 'text-green-800'}`}>
        {message.text}
      </p>
      <button
        onClick={() => onDismiss(message.id)}
        className="text-gray-400 hover:text-gray-600"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
};
