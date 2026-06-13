import React from 'react';
import { Check } from 'lucide-react';

interface Props {
  /** Активный шаг: 1 — Загрузка, 2 — Обработка, 3 — Проверка и экспорт. */
  currentStep: 1 | 2 | 3;
  /** Все шаги завершены (экран готовности) — все кружки с галочкой. */
  allComplete?: boolean;
}

const STEPS = [
  { n: 1, label: 'Загрузка' },
  { n: 2, label: 'Обработка' },
  { n: 3, label: 'Проверка и экспорт' },
] as const;

export const StepIndicator: React.FC<Props> = ({ currentStep, allComplete = false }) => {
  const stateOf = (n: number): 'done' | 'active' | 'pending' => {
    if (allComplete || n < currentStep) return 'done';
    if (n === currentStep) return 'active';
    return 'pending';
  };

  return (
    <div className="flex items-center" aria-label={`Шаг ${currentStep} из 3`}>
      {/* Десктоп: полный stepper */}
      <ol className="hidden md:flex items-center gap-1">
        {STEPS.map((step, i) => {
          const state = stateOf(step.n);
          return (
            <React.Fragment key={step.n}>
              <li className="flex items-center gap-2">
                <span
                  className={`flex items-center justify-center w-7 h-7 rounded-full text-xs font-semibold transition-colors ${
                    state === 'active'
                      ? 'bg-blue-600 text-white'
                      : state === 'done'
                        ? 'bg-green-100 text-green-600'
                        : 'bg-gray-100 text-gray-400'
                  }`}
                  aria-current={state === 'active' ? 'step' : undefined}
                >
                  {state === 'done' ? <Check className="w-4 h-4" /> : step.n}
                </span>
                <span
                  className={`text-sm whitespace-nowrap ${
                    state === 'active'
                      ? 'text-gray-900 font-medium'
                      : state === 'done'
                        ? 'text-gray-600'
                        : 'text-gray-400'
                  }`}
                >
                  {step.label}
                </span>
              </li>
              {i < STEPS.length - 1 && (
                <li
                  aria-hidden="true"
                  className={`w-8 h-0.5 mx-1 rounded-full ${
                    stateOf(step.n) === 'done' ? 'bg-green-400' : 'bg-gray-200'
                  }`}
                />
              )}
            </React.Fragment>
          );
        })}
      </ol>

      {/* Мобильный: компактная подпись */}
      <div className="md:hidden flex items-center gap-2">
        <span
          className={`flex items-center justify-center w-7 h-7 rounded-full text-xs font-semibold ${
            allComplete ? 'bg-green-100 text-green-600' : 'bg-blue-600 text-white'
          }`}
        >
          {allComplete ? <Check className="w-4 h-4" /> : currentStep}
        </span>
        <span className="text-sm font-medium text-gray-900">
          {STEPS[currentStep - 1].label}
        </span>
        <span className="text-xs text-gray-400">Шаг {currentStep} из 3</span>
      </div>
    </div>
  );
};
