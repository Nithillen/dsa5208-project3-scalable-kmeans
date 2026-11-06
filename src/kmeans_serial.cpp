#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cfloat>
#include <cstdlib>

struct Config {
    std::string input_file = "";
    std::string label_file = "data/higgs_labels.txt";
    std::string output_file = "results/centroids_serial.txt";
    size_t N = 2000000;      // Default synthetic points
    size_t d = 28;           // Default dimensions (28 for HIGGS)
    size_t K = 8;            // Number of clusters
    size_t max_iters = 40;   // Maximum iterations
    double tolerance = 1e-4; // Convergence tolerance
    unsigned int seed = 42;  // Deterministic seed
    bool use_synthetic = true;
};

// Generate synthetic Gaussian clustered data
void generate_synthetic_data(std::vector<double>& data, size_t N, size_t d, size_t K, unsigned int seed) {
    data.resize(N * d);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> center_dist(-50.0, 50.0);
    std::normal_distribution<double> noise_dist(0.0, 3.0);

    std::vector<double> true_centers(K * d);
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < d; ++j) {
            true_centers[k * d + j] = center_dist(gen);
        }
    }

    std::uniform_int_distribution<size_t> cluster_choice(0, K - 1);
    for (size_t i = 0; i < N; ++i) {
        size_t c = cluster_choice(gen);
        for (size_t j = 0; j < d; ++j) {
            data[i * d + j] = true_centers[c * d + j] + noise_dist(gen);
        }
    }
}

// High-speed CSV loader using strtod pointer scanning for multi-million row files
bool load_csv_data(const std::string& filename, std::vector<double>& data, size_t& N, size_t& d, size_t max_rows = 0) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    // Reserve a large I/O stream buffer (8 MB) for fast disk reads
    std::vector<char> io_buf(8 * 1024 * 1024);
    file.rdbuf()->pubsetbuf(io_buf.data(), io_buf.size());

    std::string line;
    data.clear();
    N = 0;
    d = 0;

    std::vector<double> row;
    row.reserve(32);

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        row.clear();
        const char* ptr = line.c_str();
        char* endptr = nullptr;
        bool valid_row = true;

        while (*ptr != '\0' && *ptr != '\r' && *ptr != '\n') {
            double val = std::strtod(ptr, &endptr);
            if (ptr == endptr) {
                valid_row = false;
                break;
            }
            row.push_back(val);
            ptr = endptr;
            while (*ptr == ',' || *ptr == ' ' || *ptr == '\t') {
                ++ptr;
            }
        }

        if (valid_row && !row.empty()) {
            if (d == 0) {
                d = row.size();
                if (max_rows > 0) {
                    data.reserve(max_rows * d);
                } else {
                    data.reserve(2000000ULL * d);
                }
            } else if (row.size() != d) {
                continue;
            }
            data.insert(data.end(), row.begin(), row.end());
            N++;
            if (max_rows > 0 && N >= max_rows) break;
        }
    }

    std::cout << "Loaded " << N << " data points with " << d << " dimensions ("
              << (N * d) << " floats) from " << filename << std::endl;
    return (N > 0 && d > 0);
}

// Compute squared Euclidean distance between point i and centroid k
inline double squared_euclidean_distance(const double* pt, const double* centroid, size_t d) {
    double dist = 0.0;
    for (size_t j = 0; j < d; ++j) {
        double diff = pt[j] - centroid[j];
        dist += diff * diff;
    }
    return dist;
}

// Deterministic initialization: pick evenly spaced points across the dataset as initial centroids
void initialize_centroids(const std::vector<double>& data, std::vector<double>& centroids, size_t N, size_t K, size_t d) {
    centroids.resize(K * d);
    size_t step = N / K;
    for (size_t k = 0; k < K; ++k) {
        size_t idx = k * step;
        for (size_t j = 0; j < d; ++j) {
            centroids[k * d + j] = data[idx * d + j];
        }
    }
}

void save_centroids(const std::string& filename, const std::vector<double>& centroids, size_t K, size_t d) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Warning: Could not save centroids to " << filename << std::endl;
        return;
    }
    out << std::setprecision(12);
    out << K << " " << d << "\n";
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < d; ++j) {
            out << centroids[k * d + j] << (j + 1 == d ? "" : " ");
        }
        out << "\n";
    }
    out.close();
    std::cout << "Final centroids saved to " << filename << " for validation.\n";
}

// Post-clustering analysis on HIGGS dataset (evaluated outside the timed benchmark loop)
void analyze_higgs_clusters(const std::string& label_file,
                            const std::vector<size_t>& assignments,
                            const std::vector<double>& centroids,
                            const std::vector<size_t>& cluster_counts,
                            size_t N, size_t K, size_t d) {
    if (d != 28) return; // Only applicable to 28-feature HIGGS dataset
    std::ifstream lfile(label_file);
    if (!lfile.is_open()) return;

    std::vector<size_t> signal_counts(K, 0);
    int lbl = 0;
    for (size_t i = 0; i < N && (lfile >> lbl); ++i) {
        if (lbl == 1) {
            signal_counts[assignments[i]]++;
        }
    }

    std::cout << "\n================ HIGGS Boson Physical Cluster Analysis ================\n";
    std::cout << "Cluster |   Events | Signal% | Lepton pT |   MET |  m_bb | m_wbb | m_wwbb\n";
    std::cout << "-----------------------------------------------------------------------\n";
    for (size_t k = 0; k < K; ++k) {
        double sig_pct = (cluster_counts[k] > 0) ? (100.0 * signal_counts[k] / cluster_counts[k]) : 0.0;
        const double* c = &centroids[k * d];
        // Feature indices in 0-indexed 28-feature HIGGS vector:
        // 0: lepton_pT, 3: missing_energy_magnitude, 25: m_bb, 26: m_wbb, 27: m_wwbb
        std::cout << "   C" << std::setw(2) << (k + 1) << "  | "
                  << std::setw(8) << cluster_counts[k] << " | "
                  << std::fixed << std::setprecision(2) << std::setw(6) << sig_pct << "% | "
                  << std::setprecision(3) << std::setw(9) << c[0] << " | "
                  << std::setw(5) << c[3] << " | "
                  << std::setw(5) << c[25] << " | "
                  << std::setw(5) << c[26] << " | "
                  << std::setw(6) << c[27] << "\n";
    }
    std::cout << "=======================================================================\n\n";
}

int main(int argc, char* argv[]) {
    Config cfg;
    size_t max_rows_override = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) { cfg.N = std::stoull(argv[++i]); max_rows_override = cfg.N; }
        else if (arg == "-d" && i + 1 < argc) cfg.d = std::stoull(argv[++i]);
        else if (arg == "-k" && i + 1 < argc) cfg.K = std::stoull(argv[++i]);
        else if (arg == "-m" && i + 1 < argc) cfg.max_iters = std::stoull(argv[++i]);
        else if (arg == "-t" && i + 1 < argc) cfg.tolerance = std::stod(argv[++i]);
        else if (arg == "-f" && i + 1 < argc) { cfg.input_file = argv[++i]; cfg.use_synthetic = false; }
        else if (arg == "-l" && i + 1 < argc) cfg.label_file = argv[++i];
        else if (arg == "-o" && i + 1 < argc) cfg.output_file = argv[++i];
    }

    std::vector<double> data;
    if (cfg.use_synthetic) {
        std::cout << "========================================================\n"
                  << "  Serial K-Means (Lloyd's Algorithm) - Synthetic Data   \n"
                  << "========================================================\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        generate_synthetic_data(data, cfg.N, cfg.d, cfg.K, cfg.seed);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Generated " << cfg.N << " x " << cfg.d << " points in "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms.\n";
    } else {
        std::cout << "========================================================\n"
                  << "  Serial K-Means (Lloyd's Algorithm) - HIGGS Dataset    \n"
                  << "========================================================\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!load_csv_data(cfg.input_file, data, cfg.N, cfg.d, max_rows_override)) {
            return 1;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "CSV load time: "
                  << std::fixed << std::setprecision(2)
                  << std::chrono::duration<double>(t1 - t0).count() << " seconds.\n";
    }

    std::cout << "Running K-Means:\n"
              << "  Points (N)    : " << cfg.N << "\n"
              << "  Features (d)  : " << cfg.d << "\n"
              << "  Total Floats  : " << (cfg.N * cfg.d) << "\n"
              << "  Clusters (K)  : " << cfg.K << "\n"
              << "  Max Iters     : " << cfg.max_iters << "\n"
              << "  Tolerance     : " << cfg.tolerance << "\n\n";

    std::vector<double> centroids;
    initialize_centroids(data, centroids, cfg.N, cfg.K, cfg.d);

    std::vector<double> new_centroids(cfg.K * cfg.d, 0.0);
    std::vector<size_t> cluster_counts(cfg.K, 0);
    std::vector<size_t> assignments(cfg.N, 0);

    auto t_start = std::chrono::high_resolution_clock::now();
    size_t iter = 0;
    double max_movement = 0.0;

    for (iter = 0; iter < cfg.max_iters; ++iter) {
        auto iter_start = std::chrono::high_resolution_clock::now();

        std::fill(new_centroids.begin(), new_centroids.end(), 0.0);
        std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

        // Expectation step: assign each point to its nearest centroid
        for (size_t i = 0; i < cfg.N; ++i) {
            const double* pt = &data[i * cfg.d];
            size_t best_cluster = 0;
            double best_dist = DBL_MAX;

            for (size_t k = 0; k < cfg.K; ++k) {
                double dist = squared_euclidean_distance(pt, &centroids[k * cfg.d], cfg.d);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cluster = k;
                }
            }

            assignments[i] = best_cluster;
            cluster_counts[best_cluster]++;
            double* new_c = &new_centroids[best_cluster * cfg.d];
            for (size_t j = 0; j < cfg.d; ++j) {
                new_c[j] += pt[j];
            }
        }

        // Maximization step: update centroids and measure displacement
        max_movement = 0.0;
        for (size_t k = 0; k < cfg.K; ++k) {
            size_t count = cluster_counts[k];
            if (count > 0) {
                double movement = 0.0;
                for (size_t j = 0; j < cfg.d; ++j) {
                    double updated_val = new_centroids[k * cfg.d + j] / count;
                    double diff = updated_val - centroids[k * cfg.d + j];
                    movement += diff * diff;
                    centroids[k * cfg.d + j] = updated_val;
                }
                movement = std::sqrt(movement);
                if (movement > max_movement) {
                    max_movement = movement;
                }
            }
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(iter_end - iter_start).count();

        std::cout << "Iteration " << std::setw(2) << (iter + 1)
                  << " | Max Centroid Shift: " << std::scientific << std::setprecision(4) << max_movement
                  << " | Time: " << std::fixed << std::setprecision(2) << iter_ms << " ms\n";

        if (max_movement < cfg.tolerance) {
            std::cout << "Converged after " << (iter + 1) << " iterations!\n";
            iter++;
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n================ Benchmark Summary ================\n"
              << "Total Iterations : " << iter << "\n"
              << "Total Time       : " << std::fixed << std::setprecision(4) << total_sec << " seconds\n"
              << "Avg Time/Iter    : " << std::fixed << std::setprecision(2) << (total_sec * 1000.0 / iter) << " ms\n"
              << "Throughput       : " << std::fixed << std::setprecision(2) << ((cfg.N * iter) / (total_sec * 1e6)) << " M points/sec\n"
              << "===================================================\n";

    if (!cfg.use_synthetic) {
        analyze_higgs_clusters(cfg.label_file, assignments, centroids, cluster_counts, cfg.N, cfg.K, cfg.d);
    }

    save_centroids(cfg.output_file, centroids, cfg.K, cfg.d);
    return 0;
}
