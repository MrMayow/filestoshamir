% Чтение последовательности из файла (одна строка с 0 и 1)
filename = '9x_1_2187.txt';
fileID = fopen(filename, 'r');
line = fgetl(fileID);
fclose(fileID);

% Преобразование строки в массив чисел 0 и 1
sequence = line - '0';

% Преобразование 0 в -1, 1 оставляем как 1
x = double(sequence)*2 - 1;

n = length(x);
% Быстрое преобразование Фурье
fft_vals = fft(x);

% Анализируем только первую половину спектра
fft_vals = fft_vals(1:floor(n/2));
freqs = (0:floor(n/2)-1)/n;

% Амплитуды (модули)
amplitudes = abs(fft_vals)/n;

% Построение спектра
figure;
plot(freqs, amplitudes);
title('Спектральный анализ псевдослучайной последовательности');
xlabel('Частота (относительная)');
ylabel('Амплитуда');
grid on;
