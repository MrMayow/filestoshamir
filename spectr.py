import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ========================================================================
# ЧТЕНИЕ БИТОВОЙ ПОСЛЕДОВАТЕЛЬНОСТИ ИЗ ФАЙЛА
# ========================================================================
def read_bit_sequence(filename):
    """Чтение битовой последовательности (0 и 1 в одну строку)"""
    try:
        with open(filename, 'r') as f:
            # Читаем весь файл
            content = f.read().strip()
            
            # Удаляем все пробелы, переводы строк и другие разделители
            content = content.replace(' ', '').replace('\n', '').replace('\r', '').replace('\t', '')
            
            # Преобразуем строку символов в массив чисел
            bit_array = np.array([int(bit) for bit in content if bit in ['0', '1']], dtype=float)
            
            return bit_array
            
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

# ========================================================================
# АВТОКОРРЕЛЯЦИЯ
# ========================================================================
def autocorrelation(x):
    """Вычисление автокорреляционной функции"""
    n = len(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[n-1:] / result[n-1]

# ========================================================================
# СПЕКТРАЛЬНЫЙ АНАЛИЗ
# ========================================================================
def spectral_analysis(signal_data):
    """FFT анализ"""
    n = len(signal_data)
    fft_values = fft(signal_data)
    fft_freq = fftfreq(n, d=1.0)
    
    # Берем только положительные частоты
    positive_idx = fft_freq > 0
    frequencies = fft_freq[positive_idx]
    amplitudes = np.abs(fft_values[positive_idx])
    
    return frequencies, amplitudes

# ========================================================================
# ОСНОВНОЙ АНАЛИЗ
# ========================================================================
def analyze_bit_sequence(filename):
    """Полный анализ битовой последовательности"""
    
    print("="*70)
    print("АНАЛИЗ БИТОВОЙ ПОСЛЕДОВАТЕЛЬНОСТИ")
    print("="*70)
    print(f"Файл: {filename}\n")
    
    # 1. ЧТЕНИЕ ДАННЫХ
    sig = read_bit_sequence(filename)
    
    if sig is None or len(sig) == 0:
        print("❌ Не удалось прочитать файл или файл пуст!")
        return None
    
    N = len(sig)
    print(f"✓ Длина последовательности: {N} бит\n")
    
    # Статистика
    zeros = np.sum(sig == 0)
    ones = np.sum(sig == 1)
    print(f"Статистика:")
    print(f"  • Нулей: {zeros} ({zeros/N*100:.2f}%)")
    print(f"  • Единиц: {ones} ({ones/N*100:.2f}%)")
    
    # 2. СПЕКТРАЛЬНЫЙ АНАЛИЗ
    print(f"\n--- Спектральный анализ (FFT) ---")
    frequencies, amplitudes = spectral_analysis(sig)
    
    # Подсчет элементов выше порога 0.95
    max_amp = np.max(amplitudes)
    threshold = 0.95 * max_amp
    count_above = int(np.sum(amplitudes > threshold))
    
    print(f"  • Максимальная амплитуда: {max_amp:.2f}")
    print(f"  • Порог (0.95 от макс.): {threshold:.2f}")
    print(f"  • >>> Элементов > 0.95 от максимума: {count_above} <<<")
    
    # 3. АВТОКОРРЕЛЯЦИЯ
    print(f"\n--- Автокорреляционный анализ ---")
    autocorr = autocorrelation(sig)
    
    print(f"  • Автокорреляция при лаге 0: {autocorr[0]:.4f}")
    print(f"  • Минимальное значение: {np.min(autocorr):.4f}")
    print(f"  • Максимальное значение: {np.max(autocorr):.4f}")
    
    # 4. POLYFIT
    print(f"\n--- Polyfit анализ спектра ---")
    n_fit = min(1000, len(frequencies))
    log_spectrum = np.log10(amplitudes[:n_fit] + 1e-10)
    poly_degree = 5
    coeffs = np.polyfit(frequencies[:n_fit], log_spectrum, poly_degree)
    poly_fit = np.polyval(coeffs, frequencies[:n_fit])
    
    print(f"  • Степень полинома: {poly_degree}")
    print(f"  • Коэффициенты: {coeffs}")
    
    # 5. ВИЗУАЛИЗАЦИЯ
    print(f"\n--- Создание графиков ---")
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Анализ битовой последовательности\n'
                 f'N = {N}, Элементов спектра > 0.95: {count_above}',
                 fontsize=14, fontweight='bold')
    
    # График 1: Исходная последовательность (первые точки)
    ax1 = plt.subplot(3, 2, 1)
    display_points = min(1000, N)
    t = np.arange(display_points)
    plt.plot(t, sig[:display_points], 'b-', linewidth=0.8, marker='o', markersize=2)
    plt.title(f'Битовая последовательность (первые {display_points} бит)', fontsize=11)
    plt.xlabel('Позиция')
    plt.ylabel('Значение')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    # График 2: Спектр
    ax2 = plt.subplot(3, 2, 2)
    plt.plot(frequencies, amplitudes, 'r-', linewidth=0.6, alpha=0.7)
    plt.axhline(y=threshold, color='g', linestyle='--', linewidth=2,
                label=f'Порог 0.95 (n={count_above})')
    plt.title('Спектр сигнала (FFT)', fontsize=11)
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 3: Спектр с Polyfit (лог. шкала)
    ax3 = plt.subplot(3, 2, 3)
    plt.semilogy(frequencies[:n_fit], amplitudes[:n_fit], 'b-', 
                 alpha=0.5, label='Спектр', linewidth=0.8)
    plt.semilogy(frequencies[:n_fit], 10**poly_fit, 'r-', 
                 label=f'Polyfit (deg={poly_degree})', linewidth=2)
    plt.title('Спектральный анализ с Polyfit', fontsize=11)
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда (лог. шкала)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 4: Автокорреляционная функция
    ax4 = plt.subplot(3, 2, 4)
    max_lag_display = min(100000, len(autocorr))
    lags = np.arange(max_lag_display)
    plt.plot(lags, autocorr[:max_lag_display], 'g-', linewidth=1)
    plt.title(f'Автокорреляционная функция ({max_lag_display} лагов)', fontsize=11)
    plt.xlabel('Лаг')
    plt.ylabel('Коэффициент автокорреляции')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # График 5: Выделение пиков спектра
    ax5 = plt.subplot(3, 2, 5)
    plt.plot(frequencies, amplitudes, 'b-', alpha=0.3, linewidth=0.5)
    high_idx = amplitudes > threshold
    if np.sum(high_idx) > 0:
        plt.scatter(frequencies[high_idx], amplitudes[high_idx],
                   c='red', s=20, marker='o', label=f'Пики > 0.95 (n={count_above})', zorder=5)
    plt.axhline(y=threshold, color='g', linestyle='--', linewidth=2, label='Порог 0.95')
    plt.title('Спектральные пики выше 0.95 от максимума', fontsize=11)
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 6: Информационная панель
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    info_text = f"""
╔═══════════════════════════════════════════════╗
║       РЕЗУЛЬТАТЫ АНАЛИЗА                      ║
╚═══════════════════════════════════════════════╝

📁 Файл: {filename}

📊 Параметры последовательности:
   • Длина: {N} бит
   • Нулей: {zeros} ({zeros/N*100:.2f}%)
   • Единиц: {ones} ({ones/N*100:.2f}%)
   • Среднее: {np.mean(sig):.4f}
   • Ст. откл.: {np.std(sig):.4f}

🎵 Спектральный анализ:
   • Макс. амплитуда: {max_amp:.2f}
   • Порог (0.95): {threshold:.2f}
   • ⚠️ Элементов > 0.95: {count_above}

🔄 Автокорреляция:
   • При лаге 0: {autocorr[0]:.4f}
   • Минимум: {np.min(autocorr):.4f}
   • Максимум: {np.max(autocorr):.4f}

📈 Polyfit:
   • Степень: {poly_degree}
   • Коэфф. [0-2]: 
     {coeffs[0]:.2e}
     {coeffs[1]:.2e}
     {coeffs[2]:.2e}
    """
    
    ax6.text(0.05, 0.5, info_text, fontsize=9, verticalalignment='center',
             family='monospace', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
    
    plt.tight_layout()
    
    # Сохранение
    import os
    output_file = f'analysis_{os.path.splitext(os.path.basename(filename))[0]}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ График сохранен: {output_file}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("="*70)
    
    return {
        'length': N,
        'zeros': zeros,
        'ones': ones,
        'max_amplitude': max_amp,
        'threshold': threshold,
        'count_above_threshold': count_above,
        'autocorr_at_0': autocorr[0]
    }

# ========================================================================
# ЗАПУСК
# ========================================================================
if __name__ == "__main__":
    # УКАЖИТЕ ИМЯ ВАШЕГО ФАЙЛА ЗДЕСЬ:
    filename = "filestoshamir/9x_8_17496.txt"
    
    # Запуск анализа
    results = analyze_bit_sequence(filename)
