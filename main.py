import os
import sys

import numpy as np
from math import sqrt
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QApplication,
    QMainWindow,
    QGridLayout,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem
)


class MainWindow(QMainWindow):
    """Главное/-ый окно/виджет приложения"""

    def __init__(self):
        """Конструктор класса MainWindow (главного окна/виджета приложения)"""
        super().__init__()

        self.directory = None

        # Инициализация массивов и векторов сырых данных
        self.input_array = None # Массив с входными данными из файлов
        self.unique_points = None # Вектор уникальных наименований пунктов
        self.fix_pos = None # Массив уникальных наименований опорных пунктов и их координат
        self.approx = None # Массив с уникальными наименованиями пунктов и их приближенными координтами
        self.all_approx = None # Массив approx с заменой координат опорных точек
        self.find_approx = None # Массив approx без опорных точек

        # Инициализация ковариационных и весовых матриц
        self.std_elements = None
        self.cov_elements = None
        self.arr_p = None

        # Инициализация счетных переменных
        self.num_bl = None # Число базовых линий
        self.num_points = None # Число пунктов
        self.num_find = None # Число искомых пунктов

        # Инициализация массивов и векторов процесса уравнивания
        self.eq_x = None # Вектор уравненных координат пунктов
        self.eq_v = None # Вектор уравненных измерений (компонентов БЛ)
        self.m = None # СКП/СКО/Стандартное отклонение измерений

        # Установка названия главного окна приложения и изменение его размера
        self.setWindowTitle('Equalization for RTKLIB')
        self.resize(600, 300)

        # Создание и настройка пустой таблицы для данных пунктов
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['Point', 'Fixed', 'X', 'Y', 'Z'])
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 50)

        # Создание кнопки "Import"
        self.import_button = QPushButton('Import')
        # Связь сигнала "clicked" со слотом "import_files" в объекте reader
        self.import_button.clicked.connect(self.import_files)

        # Создание чекбокса для учета/неучета ковариации
        self.cov_checkbox = QCheckBox('Covariation')

        # Создание кнопки "Adjust" и надписи "result_label"
        self.adjust_button = QPushButton('Equalize')
        self.adjust_button.clicked.connect(self.adjustment)
        self.label = QLabel('Import *.pos files from the selected directory')
        # Выравнивание result_label по верху
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Установка флага отвечающего за перенос слов на другую строку
        self.label.setWordWrap(True)
        self.label.setMaximumWidth(150)

        # Создание второстепенных макетов
        point_layout = QHBoxLayout()
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Добавление всех виджетов во второстепенные макеты
        point_layout.addWidget(self.table)
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.cov_checkbox)
        button_layout.addWidget(self.adjust_button)
        button_layout.addWidget(self.label)

        # Создание главного макета и его построение на основе второстепенных
        main_layout = QGridLayout()
        main_layout.addLayout(point_layout, 0, 0)
        main_layout.addLayout(button_layout, 0, 1)
        main_layout.setColumnMinimumWidth(1, 150)

        # Создание главного виджета и установка главного макета
        main_widget = QWidget()
        main_widget.setLayout(main_layout)

        # Установка центрального виджета окна
        self.setCentralWidget(main_widget)

    def import_files(self):
        """Импорт файлов из выбранной директории и формирование таблицы и матриц исходных данных"""
        self.directory = QFileDialog.getExistingDirectory(self, 'Select Folder')

        print(self.directory)

        # Выполняется, если была выбрана директория
        if self.directory:

            # Массив с именами файлов
            files = os.listdir(self.directory)
            # Создание "нулевого" массива numpy с входными данными
            self.input_array = np.empty((3 * len(files), 4), dtype=object)
            # Создание массива с исходными данными о ковариации
            self.std_elements = []
            # Счетчик количества импортированных файлов
            file_count = 0

            # Перебор каждого из файлов в директории
            for file in files:

                # Конструкция для корректного закрытия файла после извлечения необходимой информации
                with open(os.path.join(self.directory, file), 'r') as f:

                    file_count += 1
                    # Создание двух вспомогательных массивов
                    path_lines = []
                    points = []

                    # Все строки в файле записываются в переменную lines
                    lines = f.readlines()

                    # Перебор и поиск необходимых строк в файле
                    for line in lines:

                        # Поиск имен входных файлов
                        if 'inp file' in line:
                            path_lines.append(line)

                        # Поиск строки, содержащей референцные координаты (координаты БС)
                        if 'ref pos' in line:
                            split_line = line.split()
                            ref_pos = [split_line[-3], split_line[-2], split_line[-1]]

                    # Вычленение координат ровера и элементов матрицы ковариации
                    last_line = lines[-1].strip()
                    sll = last_line.split()  # split_last_line
                    measure = [sll[2], sll[3], sll[4]]

                    self.std_elements.append([sll[7], sll[8], sll[9], sll[10], sll[11], sll[12]])

                    # Получение названий БС и ровера
                    for i in range(2):
                        line = path_lines[i]
                        split_line = line.split('\\')
                        file_name = split_line[-1]
                        point_name = file_name[:4]
                        points.append(point_name.upper())

                    # Заполнение матрицы с входными данными
                    for i in range(3):
                        self.input_array[3 * (file_count - 1) + i, 0] = points[1]
                        self.input_array[3 * (file_count - 1) + i, 1] = ref_pos[i]
                        self.input_array[3 * (file_count - 1) + i, 2] = points[0]
                        self.input_array[3 * (file_count - 1) + i, 3] = measure[i]

            # Вектор уникальных названий пунктов (список всех пунктов)
            self.unique_points = np.unique(self.input_array[:, [0, 2]]).reshape(-1, 1)
            self.num_points = self.unique_points.shape[0]
            unique_pos = []

            # Вычленение приближенных координат всех пунктов (без повторов)
            for i in range(self.unique_points.shape[0]):
                found = False
                for j in range(self.input_array.shape[0]):
                    if self.input_array[j, 0] == self.unique_points[i, 0]:
                        unique_pos.append(
                            [self.input_array[j, 1], self.input_array[j + 1, 1], self.input_array[j + 2, 1]])
                        found = True
                        break
                if not found:
                    for j in range(self.input_array.shape[0]):
                        if self.input_array[j, 2] == self.unique_points[i, 0]:
                            unique_pos.append(
                                [self.input_array[j, 3], self.input_array[j + 1, 3], self.input_array[j + 2, 3]])
                            break

            unique_pos = np.array(unique_pos)
            approx = []  # Пустой массив для записи уникальных имен пунктов и их приближенных координат

            # Соединение данных из unique_points и unique_pos в матрице approx
            for i in range(self.unique_points.shape[0]):
                for j in range(unique_pos.shape[1]):
                    approx.append([self.unique_points[i, 0], unique_pos[i, j]])

            self.approx = np.array(approx)
            self.num_bl = file_count
            self.std_elements = np.array(self.std_elements)
            # Приведение всего массива к float
            self.std_elements = np.vectorize(float)(self.std_elements)

            # Вызов метода построения таблицы
            self.table_builder()

            # Замена текста у QLabel
            self.label.setText(f'Imported {file_count} files (baselines)')

    def table_builder(self):
        """Построение и настройка таблицы на основе импортированных данных"""
        # Установка количества строк для таблицы + ее очистка
        self.table.setRowCount(self.unique_points.shape[0])

        # Заполнение таблицы данными
        for row in range(self.unique_points.shape[0]):

            # Заполнение нулевого столбца названиями всех пунктов и отключение возможности их редактирования
            item = QTableWidgetItem(self.unique_points[row][0])
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item)

            # Создание, настройка отображения и установка в первый столбец виджетов чекбокса
            checkbox = QCheckBox()
            checkbox.setStyleSheet('margin-left:18;')
            self.table.setCellWidget(row, 1, checkbox)

            # Заполнение 2-4 столбцов пустыми строками для ручного ввода координат опорных точек
            for col in range(2, 5):
                item = QTableWidgetItem('')
                self.table.setItem(row, col, item)

    def adjustment(self):
        """Метод, производящий формирование необходимых матриц и уравнивание геодезической сети"""

        self.arr_p = np.eye(3 * self.num_bl)

        # Выполняется метод cov, если был отмечен чекбокс Covariation
        if self.cov_checkbox.isChecked():
            self.cov()

        self.all_approx = np.copy(self.approx)
        self.find_approx = []
        self.fix_pos = []

        # Цикл заменяет приближенные координаты опорных (отмеченных) пунктов в approx на введеные пользователем XYZ
        for i in range(self.unique_points.shape[0]):

            checkbox = self.table.cellWidget(i, 1)

            # Проверка наличия чекбокса и его состояния
            if isinstance(checkbox, QCheckBox) and checkbox.isChecked():

                # Извлечение введенных пользователем координат опорных пунктов и их имен
                items = [self.table.item(i, 0), self.table.item(i, 2), self.table.item(i, 3), self.table.item(i, 4)]
                fix = [items[0].text(), items[1].text(), items[2].text(), items[3].text()]
                self.fix_pos.append(fix)
                solid_pos = [fix[1], fix[2], fix[3]]

                # Замена значений в векторе приближенных координат всех пунктов
                for j in range(3):
                    self.all_approx[3 * i + j, 1] = solid_pos[j]

        # Создание матрицы с названиями и приближенными координатами только пунктов-роверов
        # TODO Есть проблема в том, что если приближенные и опорные координаты совпадут, то строка не будет удалена
        for i in range(self.approx.shape[0]):

            if self.approx[i, 1] == self.all_approx[i, 1]:
                self.find_approx.append(self.approx[i])

        self.find_approx = np.array(self.find_approx)
        # Установка числа искомых пунктов
        self.num_find = int(self.find_approx.shape[0] / 3)

        bl_comp = []

        # Формирование вектора с компонентами БЛ на основе массива входных данных
        for i in range(self.input_array.shape[0]):
            bl_comp.append(float(self.input_array[i, 3]) - float(self.input_array[i, 1]))

        bl_comp = np.array(bl_comp)

        # Создание матрицы коэффициентов arr_a
        arr_a = np.zeros((3 * self.num_bl, 3 * self.num_find), dtype=int)

        # Формирование матрицы коэффициентов arr_a
        for i in range(self.num_find):
            for j in range(self.num_bl):
                for k in range(3):

                    # Проверка соответствия имени искомого пункта с именами роверов в input_array
                    if self.find_approx[3 * i, 0] == self.input_array[3 * j, 2]:
                        arr_a[3 * j + k, 3 * i + k] = 1

                    # Проверка соответствия имени искомого пункта с именами БС в input_array
                    if self.find_approx[3 * i, 0] == self.input_array[3 * j, 0]:
                        arr_a[3 * j + k, 3 * i + k] = -1

        vec_bs = []
        vec_rover = []

        # Формирование векторов с приближенными координатами базовых станций и роверов в соответствии с матрице
        # коэффициентов arr_a и матрицей input_array
        for i in range(self.num_bl):

            bl = self.input_array[3 * i, 0]
            rover = self.input_array[3 * i, 2]

            for j in range(self.num_points):
                for k in range(3):

                    if bl == self.all_approx[3 * j, 0]:
                        vec_bs.append(self.all_approx[3 * j + k, 1])

                    if rover == self.all_approx[3 * j, 0]:
                        vec_rover.append(self.all_approx[3 * j + k, 1])

        # Создание массивов numpy с приведением строковых элементов к типу float
        vec_bs = np.array(list(map(float, vec_bs)))
        vec_rover = np.array(list(map(float, vec_rover)))

        # Вычисление вектора свободных членов arr_l и его транспонирование
        vec_l = vec_rover - vec_bs - bl_comp

        # Составление нормальных уравнений
        arr_r = np.dot(np.dot(arr_a.transpose(), self.arr_p), arr_a)
        vec_b = np.dot(np.dot(arr_a.transpose(), self.arr_p), vec_l)

        # Решение нормальных уравнений
        vec_x = (-1) * np.dot(np.linalg.inv(arr_r), vec_b)

        # Вычисление поправок к результатам измерений
        vec_v = np.dot(arr_a, vec_x) + vec_l

        # Вычисление уравненных координат
        pos_x = self.find_approx[:, 1].astype(float)
        self.eq_x = pos_x + vec_x

        # Вычисление уравненных измерений
        self.eq_v = bl_comp + vec_v

        # Оценка точности
        self.m = sqrt((np.dot(np.dot(vec_v.transpose(), self.arr_p), vec_v))/(3 * (self.num_bl - self.num_find)))

        # Вызов метода отвечающего за вывод результатов уравнивания
        self.export()

        # Замена текста в Qlabel
        self.label.setText('The results of the network equalization are saved in result.txt')

    def cov(self):
        """Метод работает с данными ковариации и формирует необходимые весовые матрицы"""

        self.cov_elements = np.copy(self.std_elements)

        # Восстановление ковариационной матрицы с сохранением знака '-'
        for i in range(self.num_bl):
            for j in range(6):
                if self.cov_elements[i, j] < 0:
                    self.cov_elements[i, j] = (-1) * (self.cov_elements[i, j] ** 2)
                else:
                    self.cov_elements[i, j] = self.cov_elements[i, j] ** 2

        # Элемент нормирования весовых коэффициентов
        mu = self.std_elements[0, 0]

        # Поиск наименьшего нормирующего значения mu
        for i in range(self.num_bl):
            for j in range(3):
                if self.std_elements[i, j] < mu:
                    mu = self.std_elements[i, j]

        for i in range(self.num_bl):

            sqrt_wt = np.zeros((3, 3))

            for j in range(3):
                sqrt_wt[j, j] = sqrt((mu ** 2) / self.cov_elements[i, j])

            correlation = np.eye(3)
            correlation[0, 1] = self.cov_elements[i, 3] / sqrt(self.cov_elements[i, 0] * self.cov_elements[i, 1])
            correlation[1, 2] = self.cov_elements[i, 4] / sqrt(self.cov_elements[i, 1] * self.cov_elements[i, 2])
            correlation[0, 2] = self.cov_elements[i, 5] / sqrt(self.cov_elements[i, 2] * self.cov_elements[i, 0])
            correlation[1, 0] = correlation[0, 1]
            correlation[2, 1] = correlation[1, 2]
            correlation[2, 0] = correlation[0, 2]

            bl_wt = np.dot(np.dot(sqrt_wt, correlation), sqrt_wt)

            for j in range(3):
                self.arr_p[3 * i + j, 3 * i + j] = bl_wt[j, j]

    def export(self):
        """Метод производит экспорт результатов уравнивания путем их вывода в файле формата txt"""

        exp_arr = []
        exp_arr.append(['---Fixed coordinates---', '', '', ''])
        exp_arr.append(['Point name', 'X', 'Y', 'Z'])

        for i in range(int(self.num_points - self.num_find)):
            exp_arr.append(self.fix_pos[i])

        exp_arr.append(['', '', '', ''])
        exp_arr.append(['---Equalized coordinates---', '', '', ''])
        exp_arr.append(['Point name', 'X', 'Y', 'Z',])

        for i in range(self.num_find):
            arr = [self.find_approx[3 * i, 0], self.eq_x[3 * i], self.eq_x[3 * i + 1], self.eq_x[3 * i + 2]]
            exp_arr.append(arr)

        exp_arr.append(['', '', '', ''])
        exp_arr.append(['---Equalized measurements---', '', '', ''])
        exp_arr.append(['Base-Rover name', 'X', 'Y', 'Z'])

        for i in range(self.num_bl):
            bl_name = [[self.input_array[3 * i, 0], self.input_array[3 * i, 2]]]
            bl_name = ['-'.join(row) for row in bl_name]
            arr = [bl_name[0], self.eq_v[3 * i], self.eq_v[3 * i + 1], self.eq_v[3 * i + 2]]
            exp_arr.append(arr)

        exp_arr.append(['', '', '', ''])
        exp_arr.append(['---Accuracy assessment---', '', '', ''])
        exp_arr.append(['Standard deviation', '', '', ''])
        exp_arr.append([self.m, '', '', ''])

        exp_arr = np.array(exp_arr, dtype=str)
        # Создание массива, который показывает где в массиве exp_arr находятся числовые строки
        dot_indices = np.char.find(exp_arr, '.')

        # Алгоритм округления чисел в массиве exp_arr до 4-х знаков после запятой
        for i in range(exp_arr.shape[0]):
            for j in range(exp_arr.shape[1]):
                if dot_indices[i, j] != -1:
                    num = np.round(exp_arr[i, j].astype(float), 4)
                    exp_arr[i, j] = "{:.4f}".format(num)

        # Сохранение результатов в result.txt
        np.savetxt(f'{self.directory}/result.txt', exp_arr, fmt="%-28s %-12s %-12s %-12s", delimiter="\t")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
