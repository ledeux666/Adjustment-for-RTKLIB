import os
import sys

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QApplication,
    QMainWindow,
    QGridLayout,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
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

        # Инициализация необходимых массивов и векторов
        self.input_array = None # Массив с входными данными из файлов
        self.unique_points = None # Вектор уникальных наименований пунктов
        self.approx = None # Массив с уникальными наименованиями пунктов и их приближенными координтами
        self.all_approx = None # Массив approx с заменой координат опорных точек
        self.find_approx = None # Массив approx без опорных точек

        # Инициализация необходимых переменных
        self.num_bl = None # Число базовых линий
        self.num_points = None # Число пунктов
        self.num_find = None # Число искомых пунктов

        # Установка названия главного окна приложения и изменение его размера
        self.setWindowTitle('Adjustment for RTKLIB')
        self.resize(950, 500)

        # Создание и настройка пустой таблицы для данных пунктов
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Point", "Reference", "X", "Y", "Z"])
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 70)

        # Создание кнопки "Import"
        self.import_button = QPushButton('Import')
        # Связь сигнала "clicked" со слотом "import_files" в объекте reader
        self.import_button.clicked.connect(self.import_files)

        # Создание кнопки "Adjust" и надписи "result_label"
        self.adjust_button = QPushButton('Adjust')
        self.adjust_button.clicked.connect(self.adjustment)
        self.result_label = QLabel('Result saved in...')

        # Создание второстепенных макетов
        point_layout = QHBoxLayout()
        import_layout = QHBoxLayout()
        network_layout = QHBoxLayout()
        adjustment_layout = QHBoxLayout()

        # Добавление всех виджетов во второстепенные макеты
        point_layout.addWidget(self.table)
        import_layout.addWidget(self.import_button)
        adjustment_layout.addWidget(self.adjust_button)
        adjustment_layout.addWidget(self.result_label)

        # Создание главного макета и его построение на основе второстепенных
        main_layout = QGridLayout()
        main_layout.addLayout(point_layout, 0, 0)
        main_layout.addLayout(import_layout, 1, 0)
        main_layout.addLayout(network_layout, 0, 1)
        main_layout.addLayout(adjustment_layout, 1, 1)
        # Уравнивание левых и правых виджетов в макете
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)

        # Создание главного виджета и установка главного макета
        main_widget = QWidget()
        main_widget.setLayout(main_layout)

        # Установка центрального виджета окна
        self.setCentralWidget(main_widget)

    def import_files(self):
        """Импорт файлов из выбранной директории и формирование таблицы и матриц исходных данных"""
        directory = QFileDialog.getExistingDirectory(self, "Select Folder")

        # Выполняется, если была выбрана директория
        if directory:

            # Массив с именами файлов
            files = os.listdir(directory)
            # Создание "нулевого" массива numpy с входными данными
            self.input_array = np.empty((3 * len(files), 4), dtype=object)
            # Счетчик количества импортированных файлов
            file_count = 0

            # Перебор каждого из файлов в директории
            for file in files:

                # Конструкция для корректного закрытия файла после извлечения необходимой информации
                with open(os.path.join(directory, file), 'r') as f:

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
                    # TODO Реализовать ковариацию
                    covariation = [sll[7], sll[8], sll[9], sll[10], sll[11], sll[12]]

                    # Получение названий БС и ровера
                    for i in range(2):
                        line = path_lines[i]
                        split_line = line.split("\\")
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
            # Вызов метода построения таблицы
            self.table_builder()

    def table_builder(self):
        """Построение и настройка таблицы на основе импортированных данных"""
        # Установка количества строк для таблицы + ее очистка
        self.table.setRowCount(self.unique_points.shape[0])

        # Заполнение таблицы данными
        for row in range(self.unique_points.shape[0]):

            # Заполнение нулевого столбца названиями всех пунктов и отключение возможности их редактирования
            item = QTableWidgetItem(self.unique_points[row][0])
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item)

            # Создание, настройка отображения и установка в первый столбец виджетов чекбокса
            checkbox = QCheckBox()
            checkbox.setStyleSheet("margin-left:auto; margin-right:auto;")
            self.table.setCellWidget(row, 1, checkbox)

            # Заполнение 2-4 столбцов пустыми строками для ручного ввода координат опорных точек
            for col in range(2, 5):
                item = QTableWidgetItem("")
                self.table.setItem(row, col, item)

    def adjustment(self):
        """Метод, производящий формирование необходимых матриц и уравнивание геодезической сети"""

        self.all_approx = np.copy(self.approx)
        self.find_approx = []

        # Цикл заменяет приближенные координаты опорных (отмеченных) пунктов в approx на введеные пользователем XYZ
        for i in range(self.unique_points.shape[0]):

            checkbox = self.table.cellWidget(i, 1)

            # Проверка наличия чекбокса и его состояния
            if isinstance(checkbox, QCheckBox) and checkbox.isChecked():

                # Извлечение введенных пользователем координат опорных пунктов
                items = [self.table.item(i, 2), self.table.item(i, 3), self.table.item(i, 4)]
                solid_pos = [items[0].text(), items[1].text(), items[2].text()]

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
        arr_l = vec_rover - vec_bs - bl_comp
        arr_l = arr_l.reshape(-1, 1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
