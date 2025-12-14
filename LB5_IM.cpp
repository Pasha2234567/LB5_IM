#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// ============================================================================
// КОНФИГУРАЦИЯ ПРОГРАММЫ
// ============================================================================

#define DEFAULT_DATA_FILENAME "DataV9.csv"  // Файл по умолчанию
#define SIGNIFICANCE_LEVEL 0.05            // Уровень значимости по умолчанию
#define CORRELATION_THRESHOLD 0.8          // Порог мультиколлинеарности
#define MIN_FACTORS 5                      // Минимальное число факторов
#define MIN_OBSERVATIONS 5                 // Минимальное число наблюдений

// ============================================================================
// КЛАСС ДЛЯ РАБОТЫ С ДАННЫМИ РОССТАТА
// ============================================================================

class RosstatData {
private:
    struct TimeSeries {
        string region_name;
        string region_code;
        vector<double> values;
    };

    vector<TimeSeries> series_data;
    vector<string> time_periods;
    vector<string> factor_names;
    vector<int> selected_years;  // Годы, используемые как факторы
    int target_year_idx;         // Индекс целевого года (отклика)

public:
    RosstatData() : target_year_idx(-1) {}

    // Чтение данных из CSV файла Росстата
    bool loadFromCSV(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Ошибка: не удалось открыть файл " << filename << endl;
            return false;
        }

        string line;
        int line_count = 0;
        bool years_loaded = false;

        while (getline(file, line)) {
            line_count++;

            // Пропускаем пустые строки
            if (line.empty() || line.find_first_not_of(';') == string::npos) {
                continue;
            }

            // Удаляем завершающие точки с запятой
            while (!line.empty() && line.back() == ';') {
                line.pop_back();
            }

            // Парсим строку с разделителем ';'
            vector<string> tokens;
            stringstream ss(line);
            string token;

            while (getline(ss, token, ';')) {
                tokens.push_back(token);
            }

            if (tokens.size() < 3) continue;

            // Первые строки - заголовки
            if (line_count == 1) {
                // Первая строка - названия показателей (пропускаем)
                continue;
            }
            else if (line_count == 2) {
                // Вторая строка - годы
                for (size_t i = 2; i < tokens.size(); i++) {
                    // Очищаем название года от " г."
                    string year_str = tokens[i];
                    size_t pos = year_str.find(" г.");
                    if (pos != string::npos) {
                        year_str = year_str.substr(0, pos);
                    }
                    time_periods.push_back(year_str);
                }
                years_loaded = true;
            }
            else {
                // Строки с данными
                TimeSeries ts;
                ts.region_name = tokens[0];
                ts.region_code = tokens[1];

                // Читаем значения
                for (size_t i = 2; i < tokens.size(); i++) {
                    string val_str = tokens[i];

                    // Очистка строки
                    val_str.erase(remove(val_str.begin(), val_str.end(), ' '), val_str.end());
                    val_str.erase(remove(val_str.begin(), val_str.end(), '\"'), val_str.end());

                    // Замена запятой на точку для десятичных чисел
                    size_t comma_pos = val_str.find(',');
                    if (comma_pos != string::npos) {
                        val_str[comma_pos] = '.';
                    }

                    // Удаление пробелов в качестве разделителей тысяч
                    string cleaned;
                    for (char c : val_str) {
                        if (c != ' ') cleaned += c;
                    }

                    try {
                        if (!cleaned.empty() && cleaned != "-" && cleaned != "…" &&
                            cleaned != ".." && cleaned != "\"\"" && cleaned != ".") {
                            ts.values.push_back(stod(cleaned));
                        }
                        else {
                            ts.values.push_back(numeric_limits<double>::quiet_NaN());
                        }
                    }
                    catch (...) {
                        ts.values.push_back(numeric_limits<double>::quiet_NaN());
                    }
                }

                // Добавляем ряд, если в нем достаточно данных
                int valid_count = 0;
                for (double val : ts.values) {
                    if (!isnan(val)) valid_count++;
                }

                if (valid_count >= MIN_OBSERVATIONS) {
                    series_data.push_back(ts);
                }
            }
        }

        file.close();

        if (series_data.empty()) {
            cerr << "Ошибка: не удалось загрузить данные из файла." << endl;
            return false;
        }

        cout << "✓ Загружено временных рядов: " << series_data.size() << endl;
        cout << "✓ Периодов данных: " << time_periods.size() << endl;

        if (!series_data.empty()) {
            cout << "✓ Пример региона: " << series_data[0].region_name << endl;
            cout << "✓ Количество наблюдений в ряду: " << series_data[0].values.size() << endl;
        }

        return true;
    }

    // Подготовка данных для регрессионного анализа
    // Используем несколько предыдущих лет как факторы, последний год как отклик
    bool prepareRegressionData(MatrixXd& X, VectorXd& y, int num_factors = MIN_FACTORS) {
        vector<vector<double>> X_vec;
        vector<double> y_vec;

        // Используем последний год как отклик
        target_year_idx = time_periods.size() - 1;

        // Используем предыдущие годы как факторы
        for (int i = 0; i < num_factors; i++) {
            selected_years.push_back(target_year_idx - i - 1);
            factor_names.push_back("Year_" + time_periods[target_year_idx - i - 1]);
        }

        // Для каждого региона собираем данные
        for (const auto& ts : series_data) {
            // Проверяем, есть ли данные за нужные годы
            bool valid = true;
            vector<double> x_row;

            // Данные за факторные годы
            for (int year_idx : selected_years) {
                if (year_idx >= 0 && year_idx < (int)ts.values.size() &&
                    !isnan(ts.values[year_idx])) {
                    x_row.push_back(ts.values[year_idx]);
                }
                else {
                    valid = false;
                    break;
                }
            }

            // Данные за целевой год
            if (valid && target_year_idx < (int)ts.values.size() &&
                !isnan(ts.values[target_year_idx])) {
                X_vec.push_back(x_row);
                y_vec.push_back(ts.values[target_year_idx]);
            }
        }

        if (X_vec.size() < MIN_OBSERVATIONS) {
            cerr << "Ошибка: недостаточно данных для анализа ("
                << X_vec.size() << " наблюдений)" << endl;
            return false;
        }

        // Преобразуем в матрицы Eigen
        int n = X_vec.size();
        int m = num_factors;

        X.resize(n, m);
        y.resize(n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                X(i, j) = X_vec[i][j];
            }
            y(i) = y_vec[i];
        }

        cout << "\nПодготовка данных для регрессии:" << endl;
        cout << "  Отклик (Y): данные за " << time_periods[target_year_idx] << " год" << endl;
        cout << "  Факторы (X): " << m << " предыдущих лет" << endl;
        cout << "  Наблюдений: " << n << endl;
        cout << "  Факторов: " << m << endl;

        return true;
    }

    // Получить список регионов
    vector<string> getRegionNames() const {
        vector<string> names;
        for (const auto& ts : series_data) {
            names.push_back(ts.region_name);
        }
        return names;
    }

    // Получить данные региона по индексу
    const TimeSeries& getRegionData(int idx) const {
        return series_data[idx];
    }

    // Получить названия факторов
    vector<string> getFactorNames() const {
        return factor_names;
    }

    // Получить индекс целевого года
    int getTargetYearIdx() const {
        return target_year_idx;
    }

    // Получить название целевого года
    string getTargetYear() const {
        if (target_year_idx >= 0 && target_year_idx < (int)time_periods.size()) {
            return time_periods[target_year_idx];
        }
        return "";
    }

    // Количество регионов
    int getRegionCount() const {
        return series_data.size();
    }
};

// ============================================================================
// КЛАСС МНОГОФАКТОРНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ
// ============================================================================

class MultipleLinearRegression {
private:
    VectorXd beta;              // Коэффициенты модели
    MatrixXd X_with_const;      // Матрица факторов с константой
    MatrixXd X_original;        // Оригинальная матрица факторов
    VectorXd y_original;        // Оригинальный вектор отклика
    int n, k;                   // Количество наблюдений и параметров
    vector<string> factor_names; // Названия факторов

public:
    MultipleLinearRegression() : n(0), k(0) {}

    // Обучение модели
    bool fit(const MatrixXd& X, const VectorXd& y, const vector<string>& names) {
        n = X.rows();
        k = X.cols() + 1; // +1 для константы
        factor_names = names;
        X_original = X;
        y_original = y;

        if (n <= k) {
            cerr << "  Ошибка: недостаточно наблюдений (n=" << n
                << ") для количества факторов (k=" << k << ")" << endl;
            return false;
        }

        // Добавляем столбец единиц для константы
        X_with_const.resize(n, k);
        X_with_const.col(0).setOnes();
        X_with_const.block(0, 1, n, k - 1) = X;

        // Метод наименьших квадратов
        beta = (X_with_const.transpose() * X_with_const)
            .colPivHouseholderQr()
            .solve(X_with_const.transpose() * y);

        return true;
    }

    // Прогнозирование
    VectorXd predict(const MatrixXd& X) const {
        if (X.cols() != k - 1) {
            cerr << "Ошибка: неверное количество факторов для прогноза" << endl;
            return VectorXd();
        }

        int n_new = X.rows();
        MatrixXd X_new_with_const(n_new, k);
        X_new_with_const.col(0).setOnes();
        X_new_with_const.block(0, 1, n_new, k - 1) = X;

        return X_new_with_const * beta;
    }

    // Получить коэффициенты
    VectorXd getCoefficients() const {
        return beta;
    }

    // Рассчитать статистики модели
    void calculateStatistics(double& r2, double& adj_r2, double& rmse,
        double& mape, double& mae, VectorXd& std_errors,
        VectorXd& t_stats, VectorXd& p_values) {
        VectorXd y_pred = predict(X_original);
        VectorXd residuals = y_original - y_pred;

        // Суммы квадратов
        double y_mean = y_original.mean();
        double sst = (y_original.array() - y_mean).square().sum();
        double sse = residuals.squaredNorm();
        double ssr = sst - sse;

        // R²
        r2 = (sst > 0) ? (ssr / sst) : 0;

        // Скорректированный R²
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k);

        // RMSE
        rmse = sqrt(sse / n);

        // MAE
        mae = residuals.array().abs().sum() / n;

        // MAPE
        mape = 0;
        int valid_count = 0;
        for (int i = 0; i < n; i++) {
            if (y_original[i] != 0) {
                mape += abs(residuals[i] / y_original[i]);
                valid_count++;
            }
        }
        mape = (valid_count > 0) ? (mape / valid_count * 100) : 0;

        // Стандартные ошибки коэффициентов
        MatrixXd XtX_inv = (X_with_const.transpose() * X_with_const).inverse();
        double sigma2 = sse / (n - k);
        std_errors = (sigma2 * XtX_inv.diagonal()).array().sqrt();

        // t-статистики и p-значения
        t_stats.resize(k);
        p_values.resize(k);

        for (int i = 0; i < k; i++) {
            t_stats[i] = beta[i] / std_errors[i];
            // Двусторонний t-тест
            double t_abs = abs(t_stats[i]);
            // Аппроксимация p-value через распределение Стьюдента
            p_values[i] = 2 * (1 - 0.5 * (1 + erf(t_abs / sqrt(2))));
        }
    }

    // F-статистика
    double calculateFStatistic() {
        VectorXd y_pred = predict(X_original);
        VectorXd residuals = y_original - y_pred;

        double sse = residuals.squaredNorm();
        double ssr = (y_pred.array() - y_original.mean()).square().sum();

        return (ssr / (k - 1)) / (sse / (n - k));
    }

    // Матрица корреляций между факторами
    MatrixXd calculateCorrelationMatrix() {
        int m = X_original.cols();
        MatrixXd corr = MatrixXd::Zero(m, m);

        for (int i = 0; i < m; i++) {
            for (int j = i; j < m; j++) {
                VectorXd col_i = X_original.col(i);
                VectorXd col_j = X_original.col(j);

                double mean_i = col_i.mean();
                double mean_j = col_j.mean();

                double numerator = ((col_i.array() - mean_i) * (col_j.array() - mean_j)).sum();
                double denom_i = sqrt((col_i.array() - mean_i).square().sum());
                double denom_j = sqrt((col_j.array() - mean_j).square().sum());

                if (denom_i > 0 && denom_j > 0) {
                    corr(i, j) = numerator / (denom_i * denom_j);
                    corr(j, i) = corr(i, j);
                }
            }
        }

        return corr;
    }

    // Корреляции факторов с откликом
    VectorXd calculateCorrelationWithResponse() {
        int m = X_original.cols();
        VectorXd corr_y(m);

        for (int i = 0; i < m; i++) {
            VectorXd col_i = X_original.col(i);
            double mean_i = col_i.mean();
            double mean_y = y_original.mean();

            double numerator = ((col_i.array() - mean_i) * (y_original.array() - mean_y)).sum();
            double denom_i = sqrt((col_i.array() - mean_i).square().sum());
            double denom_y = sqrt((y_original.array() - mean_y).square().sum());

            if (denom_i > 0 && denom_y > 0) {
                corr_y[i] = numerator / (denom_i * denom_y);
            }
            else {
                corr_y[i] = 0;
            }
        }

        return corr_y;
    }

    // Отбор значимых факторов по p-value
    vector<int> selectSignificantFactors(const VectorXd& p_values, double alpha) {
        vector<int> significant;
        // Начинаем с 1, пропускаем константу
        for (int i = 1; i < p_values.size(); i++) {
            if (p_values[i] < alpha) {
                significant.push_back(i - 1);
            }
        }
        return significant;
    }

    // Отбор факторов по мультиколлинеарности
    vector<int> checkMulticollinearity(double threshold) {
        MatrixXd corr_matrix = calculateCorrelationMatrix();
        vector<int> to_remove;

        for (int i = 0; i < corr_matrix.rows(); i++) {
            for (int j = i + 1; j < corr_matrix.cols(); j++) {
                if (abs(corr_matrix(i, j)) > threshold) {
                    // Удаляем фактор с меньшей корреляцией с откликом
                    VectorXd corr_y = calculateCorrelationWithResponse();
                    if (abs(corr_y[i]) < abs(corr_y[j])) {
                        if (find(to_remove.begin(), to_remove.end(), i) == to_remove.end()) {
                            to_remove.push_back(i);
                        }
                    }
                    else {
                        if (find(to_remove.begin(), to_remove.end(), j) == to_remove.end()) {
                            to_remove.push_back(j);
                        }
                    }
                }
            }
        }

        return to_remove;
    }

    // Создание новой модели с удаленными факторами
    MultipleLinearRegression removeFactors(const vector<int>& indices_to_remove) {
        if (indices_to_remove.empty()) return *this;

        // Создаем новую матрицу факторов без удаленных столбцов
        int new_cols = X_original.cols() - indices_to_remove.size();
        MatrixXd X_new(n, new_cols);

        vector<string> new_names;
        int new_idx = 0;
        for (int i = 0; i < X_original.cols(); i++) {
            if (find(indices_to_remove.begin(), indices_to_remove.end(), i) == indices_to_remove.end()) {
                X_new.col(new_idx) = X_original.col(i);
                new_names.push_back(factor_names[i]);
                new_idx++;
            }
        }

        MultipleLinearRegression new_model;
        new_model.fit(X_new, y_original, new_names);

        return new_model;
    }

    // Получить названия факторов
    vector<string> getFactorNames() const {
        return factor_names;
    }
};

// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================================

// Вывод матрицы корреляций
void printCorrelationMatrix(const MatrixXd& corr_matrix, const vector<string>& names) {
    cout << "\nМатрица корреляций между факторами:\n";
    cout << "     ";
    for (const auto& name : names) {
        cout << setw(10) << left << name.substr(0, 8) << " ";
    }
    cout << endl;

    for (int i = 0; i < corr_matrix.rows(); i++) {
        cout << setw(8) << left << names[i].substr(0, 8) << " ";
        for (int j = 0; j < corr_matrix.cols(); j++) {
            cout << setw(10) << fixed << setprecision(3) << corr_matrix(i, j) << " ";
        }
        cout << endl;
    }
}

// Вывод корреляций с откликом
void printCorrelationWithResponse(const VectorXd& corr_y, const vector<string>& names) {
    cout << "\nКорреляции факторов с откликом:\n";
    for (int i = 0; i < corr_y.size(); i++) {
        cout << "  " << setw(15) << left << names[i] << ": "
            << fixed << setprecision(4) << corr_y[i];
        if (abs(corr_y[i]) > 0.7) cout << " (сильная)";
        else if (abs(corr_y[i]) > 0.3) cout << " (умеренная)";
        else cout << " (слабая)";
        cout << endl;
    }
}

// Интерактивный отбор факторов
void interactiveFactorSelection(MultipleLinearRegression& model, double significance_level) {
    cout << "\n=== ИНТЕРАКТИВНЫЙ ОТБОР ФАКТОРОВ ===\n";

    // 1. Рассчитываем статистики
    double r2, adj_r2, rmse, mape, mae;
    VectorXd std_errors, t_stats, p_values;
    model.calculateStatistics(r2, adj_r2, rmse, mape, mae, std_errors, t_stats, p_values);

    // 2. Показываем текущие результаты
    cout << "\nТекущая модель:\n";
    cout << "  R²: " << fixed << setprecision(4) << r2 << endl;
    cout << "  Скорректированный R²: " << adj_r2 << endl;
    cout << "  Количество факторов: " << model.getFactorNames().size() << endl;

    // 3. Показываем значимые факторы
    vector<int> significant = model.selectSignificantFactors(p_values, significance_level);
    cout << "\nЗначимые факторы (p < " << significance_level << "):\n";
    if (significant.empty()) {
        cout << "  Нет значимых факторов\n";
    }
    else {
        for (int idx : significant) {
            cout << "  ✓ " << model.getFactorNames()[idx]
                << " (p = " << scientific << setprecision(2) << p_values[idx + 1] << ")\n";
        }
    }

    // 4. Проверяем мультиколлинеарность
    cout << "\nПроверка мультиколлинеарности (порог = " << CORRELATION_THRESHOLD << "):\n";
    MatrixXd corr_matrix = model.calculateCorrelationMatrix();
    vector<int> multicollinear = model.checkMulticollinearity(CORRELATION_THRESHOLD);

    if (multicollinear.empty()) {
        cout << "  Мультиколлинеарность не обнаружена\n";
    }
    else {
        cout << "  Обнаружена мультиколлинеарность у факторов:\n";
        for (int idx : multicollinear) {
            cout << "  - " << model.getFactorNames()[idx] << endl;
        }
    }

    // 5. Показываем корреляции с откликом
    VectorXd corr_y = model.calculateCorrelationWithResponse();
    printCorrelationWithResponse(corr_y, model.getFactorNames());
}

// Сохранение результатов в файл
void saveResultsToFile(const string& filename,
    const VectorXd& coefficients,
    const vector<string>& factor_names,
    double r2, double adj_r2, double rmse,
    double mape, double mae, double f_stat,
    const VectorXd& p_values,
    const vector<int>& significant_factors,
    const MatrixXd& corr_matrix,
    const VectorXd& corr_y) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Не удалось создать файл для сохранения результатов." << endl;
        return;
    }

    file << fixed << setprecision(6);
    file << "РЕЗУЛЬТАТЫ РЕГРЕССИОННОГО АНАЛИЗА\n";
    file << "=================================\n\n";

    file << "Коэффициенты модели:\n";
    file << "Константа: " << coefficients[0] << " (p-value: " << p_values[0] << ")\n";
    for (size_t i = 0; i < factor_names.size(); i++) {
        file << factor_names[i] << ": " << coefficients[i + 1]
            << " (p-value: " << p_values[i + 1] << ")";
        bool is_sig = false;
        for (int idx : significant_factors) {
            if (idx == (int)i) {
                is_sig = true;
                break;
            }
        }
        if (is_sig) file << " *ЗНАЧИМ*";
        file << "\n";
    }

    file << "\nКачество модели:\n";
    file << "R²: " << r2 << "\n";
    file << "R² скорректированный: " << adj_r2 << "\n";
    file << "F-статистика: " << f_stat << "\n";
    file << "RMSE: " << rmse << "\n";
    file << "MAE: " << mae << "\n";
    file << "MAPE: " << mape << "%\n";

    file << "\nЗначимые факторы:\n";
    if (significant_factors.empty()) {
        file << "Нет значимых факторов\n";
    }
    else {
        for (int idx : significant_factors) {
            file << factor_names[idx] << " (p = " << p_values[idx + 1] << ")\n";
        }
    }

    file << "\nКорреляции факторов с откликом:\n";
    for (size_t i = 0; i < factor_names.size(); i++) {
        file << factor_names[i] << ": " << corr_y[i] << "\n";
    }

    file << "\nМатрица корреляций между факторами:\n";
    for (size_t i = 0; i < factor_names.size(); i++) {
        file << setw(15) << left << factor_names[i];
        for (size_t j = 0; j < factor_names.size(); j++) {
            file << setw(10) << corr_matrix(i, j) << " ";
        }
        file << "\n";
    }

    file.close();
    cout << "✓ Результаты сохранены в файл: " << filename << endl;
}

// Прогноз для новых данных
void makePredictions(const RosstatData& data,
    const MultipleLinearRegression& model,
    const VectorXd& coefficients) {
    cout << "\n=== ПРОГНОЗ ДЛЯ РЕГИОНОВ ===\n";

    vector<string> regions = data.getRegionNames();
    vector<pair<double, string>> predictions;

    // Прогноз для каждого региона
    for (int i = 0; i < min(20, (int)regions.size()); i++) {
        auto region_data = data.getRegionData(i);
        vector<string> factor_names = model.getFactorNames();
        int num_factors = factor_names.size();

        // Проверяем, есть ли данные за нужные годы
        bool valid = true;
        VectorXd x_row(num_factors);

        // Предполагаем, что факторы - это последние num_factors лет перед целевым годом
        int target_idx = data.getTargetYearIdx();

        for (int j = 0; j < num_factors; j++) {
            int year_idx = target_idx - num_factors + j;
            if (year_idx >= 0 && year_idx < (int)region_data.values.size() &&
                !isnan(region_data.values[year_idx])) {
                x_row(j) = region_data.values[year_idx];
            }
            else {
                valid = false;
                break;
            }
        }

        if (valid) {
            // Прогноз
            MatrixXd X_pred(1, num_factors);
            X_pred.row(0) = x_row;
            VectorXd y_pred = model.predict(X_pred);

            if (y_pred.size() > 0) {
                predictions.push_back({ y_pred(0), region_data.region_name });
            }
        }
    }

    // Сортировка по убыванию прогноза
    sort(predictions.begin(), predictions.end(),
        [](const pair<double, string>& a, const pair<double, string>& b) {
            return a.first > b.first;
        });

    cout << "Топ-10 регионов по прогнозу:\n";
    for (int i = 0; i < min(10, (int)predictions.size()); i++) {
        cout << i + 1 << ". " << predictions[i].second
            << ": " << fixed << setprecision(2) << predictions[i].first << endl;
    }
}

// ============================================================================
// ГЛАВНАЯ ФУНКЦИЯ
// ============================================================================

int main() {
    setlocale(LC_ALL, "Russian");

    cout << "================================================" << endl;
    cout << "АНАЛИЗ ДАННЫХ РОССТАТА - МНОГОФАКТОРНАЯ РЕГРЕССИЯ" << endl;
    cout << "================================================" << endl << endl;

    try {
        // 1. ВЫБОР ФАЙЛА С ДАННЫМИ
        cout << "Доступные файлы данных:\n";
        cout << " DataV9.csv\n";

        string filename;
        cout << "\nВведите имя файла ";
        getline(cin, filename);

        if (filename.empty()) {
            filename = DEFAULT_DATA_FILENAME;
        }

        // 2. ЗАГРУЗКА ДАННЫХ
        cout << "\n1. ЗАГРУЗКА ДАННЫХ" << endl;
        cout << "------------------" << endl;

        RosstatData data;
        if (!data.loadFromCSV(filename)) {
            cerr << "\nОшибка загрузки данных. Проверьте:\n";
            cerr << "1. Наличие файла " << filename << endl;
            cerr << "2. Формат файла (CSV с разделителем ';')\n";
            cerr << "3. Структуру данных (регион, код, годы...)\n";
            return 1;
        }

        // 3. ПОДГОТОВКА ДАННЫХ
        cout << "\n2. ПОДГОТОВКА ДАННЫХ" << endl;
        cout << "--------------------" << endl;

        MatrixXd X;
        VectorXd y;

        int num_factors = MIN_FACTORS;
        cout << "Введите количество факторов для анализа (по умолчанию "
            << MIN_FACTORS << "): ";
        string input;
        getline(cin, input);
        if (!input.empty()) {
            try {
                num_factors = stoi(input);
                if (num_factors < 2) num_factors = 2;
                if (num_factors > 10) num_factors = 10;
            }
            catch (...) {
                num_factors = MIN_FACTORS;
            }
        }

        if (!data.prepareRegressionData(X, y, num_factors)) {
            cerr << "Ошибка подготовки данных для регрессии." << endl;
            return 1;
        }

        // 4. ВВОД ПАРАМЕТРОВ АНАЛИЗА
        cout << "\n3. НАСТРОЙКА ПАРАМЕТРОВ" << endl;
        cout << "----------------------" << endl;

        double significance_level = SIGNIFICANCE_LEVEL;
        cout << "Введите уровень значимости (по умолчанию "
            << SIGNIFICANCE_LEVEL << "): ";
        getline(cin, input);
        if (!input.empty()) {
            try {
                significance_level = stod(input);
                if (significance_level <= 0) significance_level = SIGNIFICANCE_LEVEL;
                if (significance_level >= 1) significance_level = SIGNIFICANCE_LEVEL;
            }
            catch (...) {
                significance_level = SIGNIFICANCE_LEVEL;
            }
        }

        // 5. ОБУЧЕНИЕ МОДЕЛИ
        cout << "\n4. ОБУЧЕНИЕ МОДЕЛИ" << endl;
        cout << "------------------" << endl;

        vector<string> factor_names = data.getFactorNames();
        MultipleLinearRegression model;
        if (!model.fit(X, y, factor_names)) {
            cerr << "Не удалось обучить модель." << endl;
            return 1;
        }
        cout << "✓ Модель успешно обучена" << endl;

        // 6. ИНТЕРАКТИВНЫЙ ОТБОР ФАКТОРОВ
        char choice;
        do {
            interactiveFactorSelection(model, significance_level);

            cout << "\nХотите удалить незначимые факторы? (y/n): ";
            getline(cin, input);
            if (!input.empty()) choice = tolower(input[0]);

            if (choice == 'y') {
                // Рассчитываем статистики для текущей модели
                double r2, adj_r2, rmse, mape, mae;
                VectorXd std_errors, t_stats, p_values;
                model.calculateStatistics(r2, adj_r2, rmse, mape, mae, std_errors, t_stats, p_values);

                // Находим незначимые факторы
                vector<int> insignificant;
                for (int i = 1; i < p_values.size(); i++) {
                    if (p_values[i] >= significance_level) {
                        insignificant.push_back(i - 1);
                    }
                }

                if (!insignificant.empty()) {
                    cout << "\nУдаление незначимых факторов:\n";
                    for (int idx : insignificant) {
                        cout << "  - " << model.getFactorNames()[idx] << endl;
                    }

                    // Создаем новую модель без незначимых факторов
                    MultipleLinearRegression new_model = model.removeFactors(insignificant);

                    if (new_model.getFactorNames().size() > 0) {
                        model = new_model;
                        cout << "\nНовая модель создана с "
                            << model.getFactorNames().size() << " факторами.\n";
                    }
                }
                else {
                    cout << "\nВсе факторы значимы.\n";
                    break;
                }
            }
            else {
                break;
            }

            cout << "\nПродолжить отбор факторов? (y/n): ";
            getline(cin, input);
            if (!input.empty()) choice = tolower(input[0]);

        } while (choice == 'y');

        // 7. ФИНАЛЬНЫЙ АНАЛИЗ
        cout << "\n5. ФИНАЛЬНЫЙ АНАЛИЗ МОДЕЛИ" << endl;
        cout << "--------------------------" << endl;

        double r2, adj_r2, rmse, mape, mae;
        VectorXd std_errors, t_stats, p_values;
        model.calculateStatistics(r2, adj_r2, rmse, mape, mae, std_errors, t_stats, p_values);
        double f_stat = model.calculateFStatistic();

        VectorXd coefficients = model.getCoefficients();
        vector<string> final_factor_names = model.getFactorNames();
        vector<int> significant_factors = model.selectSignificantFactors(p_values, significance_level);
        MatrixXd corr_matrix = model.calculateCorrelationMatrix();
        VectorXd corr_y = model.calculateCorrelationWithResponse();

        // Вывод результатов
        cout << fixed << setprecision(4);
        cout << "\nКоэффициенты модели:\n";
        cout << "Константа: " << coefficients[0]
            << " (p-value: " << scientific << setprecision(2) << p_values[0] << ")\n";

        for (size_t i = 0; i < final_factor_names.size(); i++) {
            cout << final_factor_names[i] << ": " << fixed << setprecision(4) << coefficients[i + 1]
                << " (p-value: " << scientific << setprecision(2) << p_values[i + 1] << ")";

            bool is_sig = false;
            for (int idx : significant_factors) {
                if (idx == (int)i) {
                    is_sig = true;
                    break;
                }
            }
            if (is_sig) cout << " *ЗНАЧИМ*";
            cout << endl;
        }

        cout << "\nКачество модели:\n";
        cout << "R²: " << r2 << endl;
        cout << "R² скорректированный: " << adj_r2 << endl;
        cout << "F-статистика: " << f_stat << endl;
        cout << "RMSE: " << rmse << endl;
        cout << "MAE: " << mae << endl;
        cout << "MAPE: " << mape << "%" << endl;

        cout << "\nОценка адекватности модели:\n";
        if (f_stat > 10 && r2 > 0.7) {
            cout << "  ✓ ОТЛИЧНАЯ (F > 10, R² > 0.7)\n";
        }
        else if (f_stat > 5 && r2 > 0.5) {
            cout << "  ✓ ХОРОШАЯ (F > 5, R² > 0.5)\n";
        }
        else if (f_stat > 2 && r2 > 0.3) {
            cout << "  ✓ УДОВЛЕТВОРИТЕЛЬНАЯ (F > 2, R² > 0.3)\n";
        }
        else {
            cout << "  ✗ НИЗКАЯ (требует доработки)\n";
        }

        // 8. ПРОГНОЗИРОВАНИЕ
        cout << "\n6. ПРОГНОЗИРОВАНИЕ" << endl;
        cout << "-----------------" << endl;

        makePredictions(data, model, coefficients);

        // 9. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
        cout << "\n7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ" << endl;
        cout << "-----------------------" << endl;

        saveResultsToFile("regression_results.txt",
            coefficients, final_factor_names,
            r2, adj_r2, rmse, mape, mae, f_stat,
            p_values, significant_factors,
            corr_matrix, corr_y);

        cout << "\n================================================" << endl;
        cout << "АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!" << endl;
        cout << "================================================" << endl;

        cout << "\nКраткие результаты:\n";
        cout << "------------------\n";
        cout << "1. Использовано факторов: " << final_factor_names.size() << endl;
        cout << "2. Значимых факторов: " << significant_factors.size() << endl;
        cout << "3. Качество модели (R²): " << r2 * 100 << "%" << endl;
        cout << "4. Точность прогноза (MAPE): " << mape << "%" << endl;
        cout << "5. Результаты сохранены в файл: regression_results.txt\n";

    }
    catch (const exception& e) {
        cerr << "\n!!! КРИТИЧЕСКАЯ ОШИБКА !!!" << endl;
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}
