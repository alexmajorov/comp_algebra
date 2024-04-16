import copy 
import numpy as np

class LUdcmp:
    def __init__(self, a):
        '''Функция для построения LU-разложения данной матрицы'''

        self.n = len(a)            # Размер матрицы
        self.lu = copy.deepcopy(a) # Декомпозиция
        self.indx = [0] * self.n   # Перестановка
        TINY = 1.0e-40             # Эпсилон
        vv = [0] * self.n          # Множитель каждой строки

        # Проходимся по всем рядам, чтобы
        # определить множители всех строк
        for i in range(self.n):
            big = 0.0
            for j in range(self.n):
                temp = abs(self.lu[i][j])
                if temp > big:
                    big = temp
            # Обнаружена вырожденная матрица
            if big == 0.0:     
                raise Exception("Singular matrix in LUdcmp")
            # Сохраняем множитель
            vv[i] = 1.0 / big      
        
        for k in range(self.n):
            # Иницилизируем поиск опорного элемента
            big = 0.0    
            imax = k
        
            for i in range(k, self.n):
                temp = vv[i] * abs(self.lu[i][k])
                # Обновляем опорный элемент
                if temp > big: 
                    big = temp
                    imax = i
            
            # Меняем местами строки, если необходимо
            if k != imax:
                for j in range(self.n):
                    temp = self.lu[imax][j]
                    self.lu[imax][j] = self.lu[k][j]
                    self.lu[k][j] = temp
                # Также меняем множитель
                vv[imax] = vv[k] 
            self.indx[k] = imax
            
            if self.lu[k][k] == 0.0:
                self.lu[k][k] = TINY
        
            for i in range(k + 1, self.n):
                # Делим на опорный элемент
                temp = self.lu[i][k] / self.lu[k][k]
                self.lu[i][k] /= self.lu[k][k]
                # Производим редукцию над оставшейся субматрицей
                for j in range(k + 1, self.n):
                    self.lu[i][j] -= temp * self.lu[k][j]
    
    def solve(self, b, x):
        '''Функция для решения СЛАУ Ax=b при данном b
           с помощью LU-разложения матрицы A'''

        ii = 0
        for i in range(self.n):
            x[i] = b[i]
        
        for i in range(self.n):
            ip = self.indx[i]
            sum_ = x[ip]
            x[ip] = x[i]
            if ii != 0:
                # Делаем прямую подстановку
                for j in range(ii - 1, i):
                    sum_ -= self.lu[i][j] * x[j]
            elif sum_ != 0.0:
                ii = i + 1
            x[i] = sum_
        # Делаем обратную подстановку
        for i in range(self.n - 1, -1, -1):
            sum_ = x[i]
            for j in range(i + 1, self.n):
                sum_ -= self.lu[i][j] * x[j]
            x[i] = sum_ / self.lu[i][i]

    def residual(self, a, b, x):
        '''Функция для поиска вектора невязки'''

        r = [0] * self.n
        for i in range(self.n):
            sdp = -b[i]
            for j in range(self.n):
                sdp += a[i][j] * x[j]
            r[i] = sdp
        return r


def main():
    n = 4

    a = [[3.81, 0.28, 1.28, 0.75],
         [2.25, 1.32, 4.58, 0.49],
         [5.31, 6.38, 0.98, 1.04],
         [9.39, 2.45, 3.35, 2.28]]

    b = [1, 1, 1, 1]
    x = [0, 0, 0, 0]

    alu = LUdcmp(a)
    alu.solve(b, x)

    print("Решение:")
    for i in x:
        print(round(i, 6))
    print()

    r = alu.residual(a, b, x)
    r_norm = 0
    for i in r:
        r_norm += i * i
    r_norm = r_norm ** 0.5
    print("Норма невязки = {:.2e}".format(r_norm))
    print()

    print("Решение, полученное решателем:")
    for i in np.linalg.solve(np.array(a), np.array(b)):
        print(round(i, 6))


main()
