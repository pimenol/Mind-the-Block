"""Russian strings for the chart and table HTML that build_report.py makes.

tr() replaces the longest phrases first, so a long label is never broken by a
shorter one inside it. Short words (table cells) are matched with their tags.
"""
import re

TR = {
    # legend and series names (build_report.NAME and line-chart labels)
    'GRU (learned model)': 'GRU (обучаемая модель)',
    'Kalman filter (best possible)': 'Калман (лучший возможный)',
    'Persistence (repeat last value)': 'Persistence (повтор последнего значения)',
    'GRU, block-trained, 4x steps, 32 units': 'GRU, обучена на block, 4x шагов, 32 нейрона',
    'GRU, block, 4x training, 32 units': 'GRU, block, 4x обучения, 32 нейрона',
    'GRU, block-trained': 'GRU, обучена на block',
    'GRU, scatter-trained': 'GRU, обучена на scatter',
    'GRU, mixed-trained, real targets': 'GRU, обучена на смеси, настоящие таргеты',
    'GRU, mixed-trained': 'GRU, обучена на смеси',
    'GRU, real targets': 'GRU, настоящие таргеты',
    'Memory of scatter-trained GRU': 'Память GRU, обученной на scatter',
    'Memory of mixed-trained GRU': 'Память GRU, обученной на смеси',
    'Control: shuffled labels': 'Контроль: перемешанные метки',
    'Input window (upper limit)': 'Входное окно (верхняя граница)',
    'New linear readout': 'Новый линейный выходной слой',
    'New quadratic readout': 'Новый квадратичный выходной слой',
    'GRU, clean teacher, 48 epochs': 'GRU, чистый учитель, 48 эпох',
    'GRU, clean teacher': 'GRU, чистый учитель',
    'GRU, old windows 5x': 'GRU, старые окна 5x',
    'Scatter mask: single gaps': 'Маска scatter: одиночные пропуски',
    'Block mask: 8-step outages': 'Маска block: пропуски по 8 шагов',
    'Scatter mask': 'Маска scatter', 'Block mask': 'Маска block',
    'Trained on scatter': 'Обучена на scatter', 'Trained on block': 'Обучена на block',
    'Trained on 50/50 mix': 'Обучена на смеси 50/50',
    'Seed 9': 'Сид 9', 'seed 9': 'сид 9',
    # figure titles
    'Main result: the GRU error gap, split into information lost and model failure':
        'Главный результат: разрыв в ошибке GRU, разделённый на потерянную информацию и сбой модели',
    'Share of windows where sensor 0 is 3 or more steps old at the forecast origin':
        'Доля окон, где сенсор 0 в точке прогноза старше 3 шагов или равен 3',
    'The problem: error goes up under the block mask (RMSE, block minus scatter)':
        'Задача: ошибка растёт под маской block (RMSE, block минус scatter)',
    'H3: does training on block masks remove the gap? (RMSE, block minus scatter)':
        'H3: убирает ли обучение на масках block разрыв? (RMSE, block минус scatter)',
    'H4: which outages cost error? (RMSE, against all-scatter)':
        'H4: какие пропуски добавляют ошибку? (RMSE, относительно «всё scatter»)',
    'How much the forecast trusts the last sensor-0 value, by its age (mean of 3 seeds)':
        'Насколько прогноз доверяет последнему значению сенсора 0, по его возрасту (среднее по 3 сидам)',
    'H1: extra error on the same window, by the age of the sensor-0 value (mean of 3 seeds)':
        'H1: лишняя ошибка на том же окне, по возрасту значения сенсора 0 (среднее по 3 сидам)',
    'Outage on sensor 0 only': 'Пропуск только на сенсоре 0',
    'Outage on the 5 other sensors': 'Пропуск на 5 других сенсорах',
    'Outage on all sensors': 'Пропуск на всех сенсорах',
    '(mean of 3 seeds)': '(среднее по 3 сидам)',
    'H5: can a simple probe read the age of sensor 0 from the GRU memory? (balanced accuracy, 0.5 = guessing, mean of 3 seeds)':
        'H5: может ли простая проба прочитать возраст сенсора 0 из памяти GRU? (balanced accuracy, 0.5 = угадывание, среднее по 3 сидам)',
    'H5: trust in the sensor-0 value after it stops updating for k steps (same windows, mean of 3 seeds)':
        'H5: доверие к значению сенсора 0 после того, как оно не обновляется k шагов (те же окна, среднее по 3 сидам)',
    'H5: error on old windows (age 3+) with a new readout on the frozen GRU memory (MSE)':
        'H5: ошибка на старых окнах (возраст 3+) с новым выходным слоем на замороженной памяти GRU (MSE)',
    'H6: extra error on old windows, relative to the GRU trained on real targets (1 = no change, H6 needed 0.5 or less)':
        'H6: лишняя ошибка на старых окнах относительно GRU на настоящих таргетах (1 = без изменений, H6 требовала 0.5 или меньше)',
    'H6: trust in the last sensor-0 value by its age, for each training change (mean of 3 seeds)':
        'H6: доверие к последнему значению сенсора 0 по его возрасту, для каждого изменения обучения (среднее по 3 сидам)',
    'The two kinds of missing data (drawing, not real data)': 'Два вида пропусков (схема, не настоящие данные)',
    'Scatter and block masks on one window': 'Маски scatter и block на одном окне',
    'A stale value: forward fill repeats the last reading through the outage (drawing)':
        'Старое значение: forward fill повторяет последнее показание через весь пропуск (схема)',
    'Forward fill makes an old value look current': 'Forward fill делает старое значение похожим на текущее',
    # axis labels
    'block minus scatter, MSE (raw units squared)': 'block минус scatter, MSE (исходные единицы в квадрате)',
    'steps between outage end and forecast origin': 'шагов между концом пропуска и точкой прогноза',
    'age of the sensor-0 value at the forecast origin (steps)': 'возраст значения сенсора 0 в точке прогноза (шаги)',
    'trust (forecast change per unit)': 'доверие (сдвиг прогноза на единицу)',
    'sensor-0 age at the origin under the block mask (steps)': 'возраст сенсора 0 в точке прогноза под маской block (шаги)',
    'extra MSE': 'лишняя MSE',
    'share of windows (0.20 = 1 in 5)': 'доля окон (0.20 = 1 из 5)',
    'extra RMSE under the block mask (higher is worse)': 'лишний RMSE под маской block (больше = хуже)',
    'extra RMSE under the block mask (0 = no gap)': 'лишний RMSE под маской block (0 = нет разрыва)',
    'extra RMSE when these sensors get block outages': 'лишний RMSE, когда у этих сенсоров пропуски-блоки',
    'balanced accuracy': 'balanced accuracy',
    'k = steps since the last real sensor-0 reading': 'k = шагов с последнего настоящего показания сенсора 0',
    'MSE on block windows with sensor-0 age 3 or more': 'MSE на окнах block с возрастом сенсора 0 от 3',
    'excess over the Kalman filter, divided by the real-target excess':
        'лишняя ошибка над Калманом, делённая на лишнюю ошибку с настоящими таргетами',
    # group labels and tooltips
    'only sensor 0': 'только сенсор 0', 'only the 5 others': 'только 5 других', 'all sensors': 'все сенсоры',
    'Old or not?': 'Старое или нет?', 'How old?': 'Насколько старое?',
    'Clean teacher, 48 epochs': 'Учитель, 48 эпох', 'Clean teacher': 'Чистый учитель',
    'Teacher + noise (a = 1)': 'Учитель + шум (a = 1)', 'Old windows 5x': 'Старые окна 5x',
    'Real targets, 48 epochs': 'Таргеты, 48 эпох', 'Real targets': 'Настоящие таргеты',
    'Teacher, a = ': 'Учитель, a = ',
    'oracle gap (information lost)': 'разрыв оракула (потерянная информация)',
    'excess (model failure)': 'лишняя ошибка (сбой модели)',
    '% model failure': '% сбой модели',
    'Information lost (Kalman filter gap)': 'Потерянная информация (разрыв фильтра Калмана)',
    'Model failure (GRU gap above the Kalman gap)': 'Сбой модели (разрыв GRU сверх разрыва Калмана)',
    'forecast origin': 'точка прогноза', 'outage: no readings': 'пропуск: нет показаний',
    'target (+6)': 'таргет (+6)', 'filled value is 8 steps old': 'заполненному значению 8 шагов',
    'reading present': 'показание есть', 'reading removed': 'показание удалено',
    'true signal': 'истинный сигнал', 'reading the model sees': 'показание, которое видит модель',
    'filled value the model sees': 'заполненное значение, которое видит модель',
    '>sensor 0<': '>сенсор 0<',
    # Step 1c (E8, E9)
    'T1: share of the block-mask gap that is model failure (excess over the Kalman filter)':
        'T1: доля разрыва на маске block, которая является сбоем модели (сверх фильтра Калмана)',
    'model-failure share of the gap (0.5 = half)': 'доля сбоя модели в разрыве (0.5 = половина)',
    'T2: trust in the last sensor-0 value by its age, three models (mean of 3 seeds)':
        'T2: доверие к последнему значению сенсора 0 по его возрасту, три модели (среднее по 3 сидам)',
    'T3: what a probe reads about the sensor-0 age from the final representation (balanced accuracy, mean of 3 seeds)':
        'T3: что проба читает о возрасте сенсора 0 из итогового представления (balanced accuracy, среднее по 3 сидам)',
    'R2, real data: trust in the last value of the target station by its age (mean of 3 stations)':
        'R2, реальные данные: доверие к последнему значению целевой станции по его возрасту (среднее по 3 станциям)',
    'R1, real data: block-minus-scatter MSE gap by the age of the target station at the origin (GRU, scatter-trained)':
        'R1, реальные данные: разрыв MSE block минус scatter по возрасту целевой станции в точке прогноза (GRU, обучена на scatter)',
    'age of the target-station value at the forecast origin (hours)': 'возраст значения целевой станции в точке прогноза (часы)',
    'mean extra MSE per window in this age bin': 'средняя лишняя MSE на окно в этом интервале возраста',
    'GRU in PyTorch, Adam': 'GRU в PyTorch, Adam', 'GRU, NumPy, SGD': 'GRU, NumPy, SGD',
    'Transformer, scatter-trained': 'Трансформер, обучен на scatter', 'Transformer, mixed-trained': 'Трансформер, обучен на смеси',
    'Age-aware linear reference (ridge)': 'Линейный ориентир с учётом возраста (ridge)',
    'GRU (NumPy, SGD), scatter': 'GRU (NumPy, SGD), scatter', 'GRU (PyTorch, Adam), scatter': 'GRU (PyTorch, Adam), scatter',
    'GRU (PyTorch, Adam), mixed': 'GRU (PyTorch, Adam), смесь',
    'Transformer, scatter, age input': 'Трансформер, scatter, вход с возрастом',
    'Transformer, mixed, clean teacher': 'Трансформер, смесь, чистый учитель',
    'Transformer, scatter, 48 epochs': 'Трансформер, scatter, 48 эпох',
    'Transformer, scatter': 'Трансформер, scatter', 'Transformer, mixed': 'Трансформер, смесь',
    '<td>GRU, scatter</td>': '<td>GRU, scatter</td>', '<td>GRU, mixed</td>': '<td>GRU, смесь</td>',
    'Ridge reference': 'Ridge-ориентир',
    'age 0': 'возраст 0', 'age 1': 'возраст 1', 'age 2': 'возраст 2', 'age 3-4': 'возраст 3-4', 'age 5-8': 'возраст 5-8', 'age 9+': 'возраст 9+',
    # table row labels
    'GRU, ages zeroed': 'GRU, возраст обнулён', 'GRU, age input': 'GRU, вход с возрастом',
    'GRU, fills only': 'GRU, только заполнения', 'Kalman oracle': 'Оракул Калмана',
    'scatter-trained, age input': 'обучена на scatter, вход с возрастом',
    '<td>scatter-trained</td>': '<td>обучена на scatter</td>', '<td>mixed-trained</td>': '<td>обучена на смеси</td>',
    '<td>input window</td>': '<td>входное окно</td>', '<td>mixed, zero</td>': '<td>смесь, zero</td>',
    '<td>others</td>': '<td>другие</td>', '<td>all</td>': '<td>все</td>', '<td>s0</td>': '<td>сенсор 0</td>',
    'S0 scatter / others scatter': 'S0 scatter / другие scatter', 'S0 block / others scatter': 'S0 block / другие scatter',
    'S0 scatter / others block': 'S0 scatter / другие block', 'S0 block / others block': 'S0 block / другие block',
}

NUM = r'[-+−]?\d[\d.]*%?'
RANGE = re.compile(rf'({NUM}) to ({NUM})')


def tr(s):
  for en in sorted(TR, key=len, reverse=True):
    s = s.replace(en, TR[en])
  return RANGE.sub(r'от \1 до \2', s)
