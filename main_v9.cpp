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
#include <exception>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// ========================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ВЫВОДА
// ========================================================

void displayCorrelationMatrix(const MatrixXd& corr,
    const vector<string>& labels) {
    cout << "\nМатрица корреляций между предикторами:\n";
    cout << setw(15) << " ";
    for (const auto& lbl : labels)
        cout << setw(12) << lbl.substr(0, 10);
    cout << "\n";

    for (int i = 0; i < corr.rows(); i++) {
        cout << setw(15) << labels[i].substr(0, 10);
        for (int j = 0; j < corr.cols(); j++) {
            cout << setw(12) << fixed << setprecision(4) << corr(i, j);
        }
        cout << "\n";
    }
}

void displayResponseCorrelations(const VectorXd& corr,
    const vector<string>& labels) {
    cout << "\nКорреляции предикторов с целевой переменной:\n";
    for (int i = 0; i < corr.size(); i++) {
        cout << setw(20) << left << labels[i]
            << ": " << fixed << setprecision(4) << corr[i] << "\n";
    }
}

// ========================================================
// ФУНКЦИЯ РАСПРЕДЕЛЕНИЯ СТЬЮДЕНТА
// ========================================================

double computeStudentCDF(double t_value, int dof) {
    double x = dof / (dof + t_value * t_value);
    auto betaCF = [](double a, double b, double x_val) {
        const int MAX_ITER = 100;
        const double EPS = 3e-7;
        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x_val / qap;
        if (fabs(d) < 1e-30) d = 1e-30;
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= MAX_ITER; ++m) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x_val / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (fabs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (fabs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x_val / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (fabs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (fabs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (fabs(delta - 1.0) < EPS) break;
        }
        return h;
        };

    double a = dof / 2.0;
    double b = 0.5;
    double bt = exp(
        lgamma(a + b) - lgamma(a) - lgamma(b)
        + a * log(x) + b * log(1.0 - x)
    );

    if (t_value >= 0)
        return 1.0 - 0.5 * bt * betaCF(a, b, x);
    else
        return 0.5 * bt * betaCF(a, b, x);
}

// ========================================================
// КЛАСС МНОГОФАКТОРНОЙ РЕГРЕССИИ
// ========================================================

class MultiRegressor {
private:
    MatrixXd predictors_original;
    MatrixXd predictors_extended;
    VectorXd target_original;
    VectorXd model_params;
    int observations;
    int param_count;
    vector<string> predictor_labels;

public:
    MultiRegressor() : observations(0), param_count(0) {}

    bool train(const MatrixXd& predictors, const VectorXd& target,
        const vector<string>& labels) {
        if (predictors.rows() != target.size()) return false;
        observations = predictors.rows();
        int predictor_cnt = predictors.cols();
        param_count = predictor_cnt + 1;
        predictors_original = predictors;
        target_original = target;
        predictor_labels = labels;

        predictors_extended.resize(observations, param_count);
        predictors_extended.col(0) = VectorXd::Ones(observations);
        predictors_extended.block(0, 1, observations, predictor_cnt) = predictors;

        model_params = (predictors_extended.transpose() * predictors_extended)
            .inverse()
            * predictors_extended.transpose() * target;

        return true;
    }

    VectorXd estimate(const MatrixXd& predictors) const {
        MatrixXd extended(predictors.rows(), param_count);
        extended.col(0) = VectorXd::Ones(predictors.rows());
        extended.block(0, 1, predictors.rows(), predictors.cols()) = predictors;
        return extended * model_params;
    }

    VectorXd getParameters() const {
        return model_params;
    }

    vector<string> getPredictorLabels() const {
        return predictor_labels;
    }

    void computeMetrics(double& r_squared,
        double& adjusted_r2,
        double& error_rmse,
        double& error_mape,
        double& error_mae,
        VectorXd& std_errs,
        VectorXd& t_vals,
        VectorXd& p_vals) {
        VectorXd predictions = estimate(predictors_original);
        VectorXd errors = target_original - predictions;

        double ss_err = errors.squaredNorm();
        double ss_tot = (target_original.array() - target_original.mean()).square().sum();
        r_squared = 1.0 - ss_err / ss_tot;
        adjusted_r2 = 1.0 - (1.0 - r_squared) * (observations - 1) / (observations - param_count);
        error_rmse = sqrt(ss_err / observations);
        error_mae = errors.array().abs().mean();
        error_mape = 0.0;
        int valid_cnt = 0;

        for (int i = 0; i < observations; i++) {
            if (fabs(target_original[i]) > 1e-12) {
                error_mape += fabs(errors[i] / target_original[i]);
                valid_cnt++;
            }
        }
        if (valid_cnt > 0) error_mape = error_mape / valid_cnt * 100.0;

        MatrixXd XtX_inv = (predictors_extended.transpose() * predictors_extended).inverse();
        double sigma_sq = ss_err / (observations - param_count);
        std_errs = (sigma_sq * XtX_inv.diagonal()).array().sqrt();

        t_vals.resize(param_count);
        p_vals.resize(param_count);
        int dof = observations - param_count;

        for (int i = 0; i < param_count; i++) {
            t_vals[i] = model_params[i] / std_errs[i];
            double t_abs = fabs(t_vals[i]);
            double cdf = computeStudentCDF(t_abs, dof);
            p_vals[i] = 2.0 * (1.0 - cdf);
        }
    }

    double computeFScore() {
        VectorXd predictions = estimate(predictors_original);
        VectorXd errors = target_original - predictions;
        double ss_err = errors.squaredNorm();
        double ss_reg = (predictions.array() - target_original.mean()).square().sum();
        return (ss_reg / (param_count - 1)) / (ss_err / (observations - param_count));
    }

    MatrixXd computePredictorCorrelations() {
        int m = predictors_original.cols();
        MatrixXd corr_mat = MatrixXd::Zero(m, m);

        for (int i = 0; i < m; i++) {
            for (int j = i; j < m; j++) {
                VectorXd xi = predictors_original.col(i);
                VectorXd xj = predictors_original.col(j);
                double mi = xi.mean();
                double mj = xj.mean();
                double numerator = ((xi.array() - mi) * (xj.array() - mj)).sum();
                double di = sqrt((xi.array() - mi).square().sum());
                double dj = sqrt((xj.array() - mj).square().sum());

                if (di > 0 && dj > 0) {
                    corr_mat(i, j) = numerator / (di * dj);
                    corr_mat(j, i) = corr_mat(i, j);
                }
            }
        }
        return corr_mat;
    }

    VectorXd computeTargetCorrelations() {
        int m = predictors_original.cols();
        VectorXd corr_vec(m);
        double my = target_original.mean();
        double dy = sqrt((target_original.array() - my).square().sum());

        for (int i = 0; i < m; i++) {
            VectorXd xi = predictors_original.col(i);
            double mi = xi.mean();
            double di = sqrt((xi.array() - mi).square().sum());

            if (di > 0 && dy > 0) {
                double numerator = ((xi.array() - mi) * (target_original.array() - my)).sum();
                corr_vec[i] = numerator / (di * dy);
            }
            else {
                corr_vec[i] = 0.0;
            }
        }
        return corr_vec;
    }

    vector<int> selectImportantPredictors(const VectorXd& p_vals,
        double alpha_level) {
        vector<int> selected;
        for (int i = 1; i < p_vals.size(); i++) {
            if (p_vals[i] < alpha_level) {
                selected.push_back(i - 1);
            }
        }
        return selected;
    }

    vector<int> detectCollinearity(double collin_thresh) {
        MatrixXd corr = computePredictorCorrelations();
        VectorXd corr_target = computeTargetCorrelations();
        vector<int> to_exclude;

        for (int i = 0; i < corr.rows(); i++) {
            for (int j = i + 1; j < corr.cols(); j++) {
                if (fabs(corr(i, j)) > collin_thresh) {
                    if (fabs(corr_target[i]) < fabs(corr_target[j])) {
                        if (find(to_exclude.begin(), to_exclude.end(), i) == to_exclude.end())
                            to_exclude.push_back(i);
                    }
                    else {
                        if (find(to_exclude.begin(), to_exclude.end(), j) == to_exclude.end())
                            to_exclude.push_back(j);
                    }
                }
            }
        }
        return to_exclude;
    }

    MultiRegressor excludePredictors(const vector<int>& indices) {
        if (indices.empty()) return *this;

        int new_m = predictors_original.cols() - indices.size();
        MatrixXd X_new(observations, new_m);
        vector<string> labels_new;
        int col_idx = 0;

        for (int i = 0; i < predictors_original.cols(); i++) {
            if (find(indices.begin(), indices.end(), i) == indices.end()) {
                X_new.col(col_idx) = predictors_original.col(i);
                labels_new.push_back(predictor_labels[i]);
                col_idx++;
            }
        }

        MultiRegressor new_model;
        new_model.train(X_new, target_original, labels_new);
        return new_model;
    }
};

// ============================================================
// КЛАСС ДЛЯ РАБОТЫ С ДАННЫМИ
// ============================================================

class DataLoader {
private:
    struct SeriesEntry {
        string region_label;
        string region_id;
        vector<double> measurements;
    };

    vector<SeriesEntry> dataset;
    vector<string> time_labels;
    vector<string> predictor_names;
    vector<int> selected_indices;
    int target_index;

public:
    DataLoader() : target_index(-1) {}

    bool importCSV(const string& filepath) {
        ifstream input(filepath);
        if (!input.is_open()) {
            cerr << "Ошибка: не удалось открыть файл " << filepath << endl;
            return false;
        }

        string line_str;
        int line_num = 0;

        while (getline(input, line_str)) {
            line_num++;

            if (line_str.empty() || line_str.find_first_not_of(';') == string::npos) {
                continue;
            }

            while (!line_str.empty() && line_str.back() == ';') {
                line_str.pop_back();
            }

            vector<string> parts;
            stringstream ss(line_str);
            string token;

            while (getline(ss, token, ';')) {
                parts.push_back(token);
            }

            if (parts.size() < 3) continue;

            if (line_num == 1) {
                continue;
            }
            else if (line_num == 2) {
                for (size_t i = 2; i < parts.size(); i++) {
                    string year_str = parts[i];
                    size_t pos = year_str.find(" г.");
                    if (pos != string::npos) {
                        year_str = year_str.substr(0, pos);
                    }
                    time_labels.push_back(year_str);
                }
            }
            else {
                SeriesEntry entry;
                entry.region_label = parts[0];
                entry.region_id = parts[1];

                for (size_t i = 2; i < parts.size(); i++) {
                    string val_str = parts[i];

                    val_str.erase(remove(val_str.begin(), val_str.end(), ' '), val_str.end());
                    val_str.erase(remove(val_str.begin(), val_str.end(), '\"'), val_str.end());

                    size_t comma_pos = val_str.find(',');
                    if (comma_pos != string::npos) {
                        val_str[comma_pos] = '.';
                    }

                    string clean_val;
                    for (char ch : val_str) {
                        if (ch != ' ') clean_val += ch;
                    }

                    try {
                        if (!clean_val.empty() && clean_val != "-" && clean_val != "…" &&
                            clean_val != ".." && clean_val != "\"\"" && clean_val != ".") {
                            entry.measurements.push_back(stod(clean_val));
                        }
                        else {
                            entry.measurements.push_back(numeric_limits<double>::quiet_NaN());
                        }
                    }
                    catch (...) {
                        entry.measurements.push_back(numeric_limits<double>::quiet_NaN());
                    }
                }

                int valid_vals = 0;
                for (double v : entry.measurements) {
                    if (!isnan(v)) valid_vals++;
                }

                if (valid_vals >= 5) {
                    dataset.push_back(entry);
                }
            }
        }

        input.close();

        if (dataset.empty()) {
            cerr << "Ошибка: не удалось загрузить данные из файла." << endl;
            return false;
        }

        cout << "Загружено временных рядов: " << dataset.size() << endl;
        cout << "Периодов данных: " << time_labels.size() << endl;

        if (!dataset.empty()) {
            cout << "Пример региона: " << dataset[0].region_label << endl;
        }

        return true;
    }

    bool prepareForRegression(MatrixXd& predictors, VectorXd& target, int pred_count = 5) {
        vector<vector<double>> pred_rows;
        vector<double> target_vals;

        target_index = time_labels.size() - 1;

        predictor_names.clear();
        selected_indices.clear();

        for (int i = 0; i < pred_count; i++) {
            selected_indices.push_back(target_index - i - 1);
            predictor_names.push_back("Year_" + time_labels[target_index - i - 1]);
        }

        for (const auto& entry : dataset) {
            bool is_valid = true;
            vector<double> row_vals;

            for (int idx : selected_indices) {
                if (idx >= 0 && idx < (int)entry.measurements.size() &&
                    !isnan(entry.measurements[idx])) {
                    row_vals.push_back(entry.measurements[idx]);
                }
                else {
                    is_valid = false;
                    break;
                }
            }

            if (is_valid && target_index < (int)entry.measurements.size() &&
                !isnan(entry.measurements[target_index])) {
                pred_rows.push_back(row_vals);
                target_vals.push_back(entry.measurements[target_index]);
            }
        }

        if (pred_rows.size() < 5) {
            cerr << "Ошибка: недостаточно данных для анализа ("
                << pred_rows.size() << " наблюдений)" << endl;
            return false;
        }

        int n_obs = pred_rows.size();
        int m_pred = pred_count;

        predictors.resize(n_obs, m_pred);
        target.resize(n_obs);

        for (int i = 0; i < n_obs; i++) {
            for (int j = 0; j < m_pred; j++) {
                predictors(i, j) = pred_rows[i][j];
            }
            target(i) = target_vals[i];
        }

        cout << "\nПодготовка данных для регрессии:" << endl;
        cout << "  Целевая переменная (Y): данные за " << time_labels[target_index] << " год" << endl;
        cout << "  Предикторы (X): " << m_pred << " предыдущих лет" << endl;
        cout << "  Наблюдений: " << n_obs << endl;

        return true;
    }

    vector<string> getPredictorNames() const {
        return predictor_names;
    }

    int getTargetIndex() const {
        return target_index;
    }

    string getTargetYear() const {
        if (target_index >= 0 && target_index < (int)time_labels.size()) {
            return time_labels[target_index];
        }
        return "";
    }

    vector<string> getRegionNames() const {
        vector<string> names;
        for (const auto& entry : dataset) {
            names.push_back(entry.region_label);
        }
        return names;
    }

    const SeriesEntry& getRegionEntry(int idx) const {
        return dataset[idx];
    }
};

// ============================================================
// СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В ФАЙЛ
// ============================================================

void exportResults(const string& out_file,
    const VectorXd& coeffs,
    const vector<string>& labels,
    double r2, double adj_r2,
    double rmse, double mape, double mae,
    double f_score,
    const VectorXd& p_values) {
    ofstream out(out_file);
    if (!out.is_open()) {
        cerr << "Ошибка сохранения файла\n";
        return;
    }

    out << fixed << setprecision(6);
    out << "РЕЗУЛЬТАТЫ РЕГРЕССИОННОГО АНАЛИЗА\n";
    out << "=================================\n\n";

    out << "Коэффициенты модели:\n";
    out << "Константа: " << coeffs[0]
        << " (p = " << p_values[0] << ")\n";

    for (size_t i = 0; i < labels.size(); i++) {
        out << labels[i] << ": "
            << coeffs[i + 1]
            << " (p = " << p_values[i + 1] << ")\n";
    }

    out << "\nКачество модели:\n";
    out << "R2: " << r2 << "\n";
    out << "R2 adj: " << adj_r2 << "\n";
    out << "F-stat: " << f_score << "\n";
    out << "RMSE: " << rmse << "\n";
    out << "MAE: " << mae << "\n";
    out << "MAPE: " << mape << "%\n";

    out.close();
}

// ============================================================
// ГЛАВНАЯ ФУНКЦИЯ
// ============================================================

int main() {
    setlocale(LC_ALL, "Russian");

    cout << "=============================================\n";
    cout << "МНОГОФАКТОРНАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ\n";
    cout << "=============================================\n\n";

    try {
        // ВЫБОР ФАЙЛА С ДАННЫМИ
        cout << "Введите имя файла с данными (по умолчанию DataV9.csv): ";
        string filename;
        getline(cin, filename);

        if (filename.empty()) {
            filename = "DataV9.csv";
        }

        // ЗАГРУЗКА ДАННЫХ
        cout << "\n1. ЗАГРУЗКА ДАННЫХ\n";
        cout << "------------------\n";

        DataLoader loader;
        if (!loader.importCSV(filename)) {
            cerr << "Ошибка загрузки данных.\n";
            return 1;
        }

        // ПОДГОТОВКА ДАННЫХ
        cout << "\n2. ПОДГОТОВКА ДАННЫХ\n";
        cout << "--------------------\n";

        MatrixXd predictors;
        VectorXd target;

        int pred_count = 5;
        cout << "Введите количество предикторов (по умолчанию 5): ";
        string user_input;
        getline(cin, user_input);

        if (!user_input.empty()) {
            try {
                pred_count = stoi(user_input);
                if (pred_count < 2) pred_count = 2;
                if (pred_count > 10) pred_count = 10;
            }
            catch (...) {
                pred_count = 5;
            }
        }

        if (!loader.prepareForRegression(predictors, target, pred_count)) {
            cerr << "Ошибка подготовки данных для регрессии.\n";
            return 1;
        }

        // НАСТРОЙКА ПАРАМЕТРОВ
        cout << "\n3. НАСТРОЙКА ПАРАМЕТРОВ\n";
        cout << "----------------------\n";

        double alpha = 0.05;
        cout << "Введите уровень значимости (по умолчанию 0.05): ";
        getline(cin, user_input);

        if (!user_input.empty()) {
            try {
                alpha = stod(user_input);
                if (alpha <= 0) alpha = 0.05;
                if (alpha >= 1) alpha = 0.05;
            }
            catch (...) {
                alpha = 0.05;
            }
        }

        double collin_thresh = 0.8;
        cout << "Введите порог мультиколлинеарности (по умолчанию 0.8): ";
        getline(cin, user_input);

        if (!user_input.empty()) {
            try {
                collin_thresh = stod(user_input);
                if (collin_thresh <= 0) collin_thresh = 0.8;
                if (collin_thresh >= 1) collin_thresh = 0.8;
            }
            catch (...) {
                collin_thresh = 0.8;
            }
        }

        // ОБУЧЕНИЕ МОДЕЛИ
        cout << "\n4. ОБУЧЕНИЕ МОДЕЛИ\n";
        cout << "------------------\n";

        vector<string> pred_labels = loader.getPredictorNames();
        MultiRegressor reg_model;

        if (!reg_model.train(predictors, target, pred_labels)) {
            cerr << "Не удалось обучить модель.\n";
            return 1;
        }

        cout << "Модель успешно обучена\n";

        // РАСЧЁТ СТАТИСТИК
        double r2, adj_r2, rmse, mape, mae;
        VectorXd se, t_stats, p_vals;
        reg_model.computeMetrics(r2, adj_r2, rmse, mape, mae, se, t_stats, p_vals);
        double f_score = reg_model.computeFScore();

        // ВЫВОД РЕЗУЛЬТАТОВ
        cout << "\n5. РЕЗУЛЬТАТЫ АНАЛИЗА\n";
        cout << "--------------------\n";

        cout << "\nКоэффициенты:\n";
        VectorXd coeffs = reg_model.getParameters();
        cout << "Const: " << coeffs[0]
            << " (p=" << p_vals[0] << ")\n";

        for (size_t i = 0; i < pred_labels.size(); i++) {
            cout << pred_labels[i] << ": "
                << coeffs[i + 1]
                << " (p=" << p_vals[i + 1] << ")\n";
        }

        cout << "\nR2 = " << r2
            << "\nR2 adj = " << adj_r2
            << "\nF = " << f_score
            << "\nRMSE = " << rmse
            << "\nMAE = " << mae
            << "\nMAPE = " << mape << "%\n";

        // ПРОВЕРКА МУЛЬТИКОЛЛИНЕАРНОСТИ
        cout << "\n6. ПРОВЕРКА МУЛЬТИКОЛЛИНЕАРНОСТИ\n";
        cout << "-------------------------------\n";

        MatrixXd corr_mat = reg_model.computePredictorCorrelations();
        displayCorrelationMatrix(corr_mat, pred_labels);

        VectorXd corr_target = reg_model.computeTargetCorrelations();
        displayResponseCorrelations(corr_target, pred_labels);

        vector<int> collinear = reg_model.detectCollinearity(collin_thresh);
        if (!collinear.empty()) {
            cout << "\nОбнаружена мультиколлинеарность у предикторов:\n";
            for (int idx : collinear) {
                cout << "  - " << pred_labels[idx] << endl;
            }

            cout << "\nХотите удалить проблемные предикторы? (y/n): ";
            getline(cin, user_input);

            if (!user_input.empty() && tolower(user_input[0]) == 'y') {
                MultiRegressor updated_model = reg_model.excludePredictors(collinear);
                reg_model = updated_model;
                pred_labels = reg_model.getPredictorLabels();

                // Пересчитываем статистики
                reg_model.computeMetrics(r2, adj_r2, rmse, mape, mae, se, t_stats, p_vals);
                f_score = reg_model.computeFScore();
                coeffs = reg_model.getParameters();

                cout << "\nМодель обновлена. Новые коэффициенты:\n";
                cout << "Const: " << coeffs[0] << " (p=" << p_vals[0] << ")\n";
                for (size_t i = 0; i < pred_labels.size(); i++) {
                    cout << pred_labels[i] << ": " << coeffs[i + 1] << " (p=" << p_vals[i + 1] << ")\n";
                }
            }
        }

        // ОТБОР ЗНАЧИМЫХ ПРЕДИКТОРОВ
        cout << "\n7. ОТБОР ЗНАЧИМЫХ ПРЕДИКТОРОВ\n";
        cout << "-------------------------\n";

        vector<int> significant = reg_model.selectImportantPredictors(p_vals, alpha);
        if (!significant.empty()) {
            cout << "Значимые предикторы (p < " << alpha << "):\n";
            for (int idx : significant) {
                cout << "  ok " << pred_labels[idx]
                    << " (p = " << scientific << setprecision(2) << p_vals[idx + 1] << ")\n";
            }
        }
        else {
            cout << "Нет значимых предикторов на уровне " << alpha << "\n";
        }

        // СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
        cout << "\n8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ\n";
        cout << "-----------------------\n";

        exportResults("regression_results_v9.txt",
            coeffs, pred_labels,
            r2, adj_r2,
            rmse, mape, mae,
            f_score, p_vals);

        cout << "Анализ завершён корректно\n";
        cout << "Результаты сохранены в файл: regression_results_v9.txt\n";

    }
    catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
