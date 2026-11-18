import hashlib
import time
import math
import os

import pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import numpy.random as rnd


def createKey(n: int, Q: float):
    AliceKey = rnd.randint(0, 2, size=(n))
    error_line = np.array([1 if rnd.random() < Q else 0 for _ in range(n)])
    BobKey = AliceKey ^ error_line
    return AliceKey, BobKey


def list_to_matrix(input_list, k, n):
    if len(input_list) < k * n:
        while len(input_list) != k * n:
            input_list.append(0)
    matrix = np.array(input_list).reshape(k, n)
    return matrix


class TreeParityMachine:
    def __init__(self, K, N, L, weights):
        self.K = K
        self.N = N
        self.L = L
        self.weights = weights

    def update(self, X, tau_Partner):
        tau = self.tau
        sigma = self.sigma
        weights = self.weights
        L = self.L
        if tau == tau_Partner:
            self.hebbian(weights, X, sigma, tau, tau_Partner, L)

    def compute_output(self, X):
        X = X.reshape([self.K, self.N])
        sigma = np.where(np.sum(X * self.weights, axis=1) > 0, 1, -1)
        tau = np.prod(sigma)
        self.X = X
        self.sigma = sigma
        self.tau = tau
        return tau

    def hebbian(self, weights, X, sigma, tau_A, tau_B, l):

        mask = (sigma == tau_A).astype(weights.dtype).reshape(-1, 1)
        weights += X * tau_A * mask
        np.clip(weights, -l, l, out=weights)


def binary_to_decimal(binary_array, rangeBits):
    weightList = []
    minus = 0
    for k in range(0, rangeBits):
        minus += 2 ** k
    minus = minus // 2
    minus += 1
    for i in range(0, len(binary_array), rangeBits):
        torture = ''
        for j in range(0, rangeBits):
            torture += str(binary_array[i + j])
        a = int(torture, 2)
        a -= minus
        weightList.append(a)
    return weightList



def get_parameter_sets_for_key_length(target_key_length=8192, bits_min=1, bits_max=13):
    parameter_sets = []
    for Bits in range(bits_min, bits_max + 1):
        if target_key_length % Bits != 0:
            continue
        product_kn = target_key_length // Bits
        for K in range(1, product_kn + 1):
            if product_kn % K == 0:
                N = product_kn // K
                if K <= N:
                    parameter_sets.append([K, N, Bits])
    return parameter_sets


def _measure_chunk(args):
    (K, N, Bits, kStart, kLimit, TCounterLimit,
     measurements_chunk, max_iterations, Q, seed) = args

    # Каждый процесс должен иметь собственный, не коррелированный
    # с другими, поток случайных чисел.
    rnd.seed(seed)

    n_local = K * N * Bits
    L_local = 2 ** Bits
    L_local = L_local // 2
    M = L_local * 0.25
    if M<1:
        M = 1

    failures = 0
    sync_iterations_sum = 0
    disclosed_information_sum = 0.0
    fact_sync_iterations_sum = 0
    messages_sent_sum = 0
    traffic_volume_bits_sum = 0

    for _ in range(measurements_chunk):
        keySlice_A, keySlice_B = createKey(n_local, Q)

        weights_A = binary_to_decimal(keySlice_A, Bits)
        weights_B = binary_to_decimal(keySlice_B, Bits)

        arrayWeights_A = list_to_matrix(weights_A, K, N)
        arrayWeights_B = list_to_matrix(weights_B, K, N)

        TPM_A = TreeParityMachine(K, N, L_local, arrayWeights_A)
        TPM_B = TreeParityMachine(K, N, L_local, arrayWeights_B)

        iter_ = 0
        iterations = 0


        while iter_ < max_iterations:
            iter_ += 1
            iterations += 1
            if M> 1:
                vals = np.r_[np.arange(-M, 0), np.arange(1, M + 1)]
                X = np.random.choice(vals, size=(K, N))
            else:
                X = np.random.randint(-1, 2, [K, N])

            X = np.asarray(X, dtype=TPM_A.weights.dtype)
            out_A = TPM_A.compute_output(X)
            out_B = TPM_B.compute_output(X)

            messages_sent_sum += 2
            traffic_volume_bits_sum += 2

            if out_A == out_B:
                TPM_A.update(X, out_B)
                TPM_B.update(X, out_A)

                weights_equal = np.array_equal(
                    TPM_A.weights.flatten(), TPM_B.weights.flatten()
                )


                if weights_equal:
                    factSyncIter = iter_
                    fact_sync_iterations_sum += factSyncIter
                    sync_iterations_sum += iterations
                    disclosed_information_sum += 2 * iterations
                    break

            if iter_ == max_iterations:
                failures += 1
    return (failures, sync_iterations_sum, disclosed_information_sum,
            fact_sync_iterations_sum, messages_sent_sum,
            traffic_volume_bits_sum, measurements_chunk)




def measure_single_parameter_set_parallel(K, N, Bits, kStart=1, kLimit=5,
                                           TCounterLimit=50, measurements=10000,
                                           max_iterations=1000, Q=0.05,
                                           n_workers=None, verbose=True):
    n_workers = n_workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, measurements))


    base = measurements // n_workers
    remainder = measurements % n_workers
    chunks = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    chunks = [c for c in chunks if c > 0]


    seeds = [int.from_bytes(os.urandom(4), "little") for _ in chunks]

    tasks = [
        (K, N, Bits, kStart, kLimit, TCounterLimit, chunk, max_iterations, Q, seed)
        for chunk, seed in zip(chunks, seeds)
    ]

    total_failures = 0
    total_sync_iterations = 0
    total_disclosed_information = 0.0
    total_fact_sync_iterations = 0
    total_messages_sent = 0
    total_traffic_volume_bits = 0
    total_measurements = 0

    # t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        futures = [ex.submit(_measure_chunk, t) for t in tasks]
        for f in as_completed(futures):
            (failures, sync_it, disc_info, fact_sync_it,
             messages_sent, traffic_volume_bits, m) = f.result()
            total_failures += failures
            total_sync_iterations += sync_it
            total_disclosed_information += disc_info
            total_fact_sync_iterations += fact_sync_it
            total_messages_sent += messages_sent
            total_traffic_volume_bits += traffic_volume_bits
            total_measurements += m

    # elapsed = time.perf_counter() - t_start

    fer_average = total_failures / total_measurements

    denom = total_measurements - (total_failures)
    if denom > 0:
        fact_sync_iterations_average = total_fact_sync_iterations / denom
        sync_iterations_average = total_sync_iterations / denom
        disclosed_information_average = total_disclosed_information / denom
    else:
        fact_sync_iterations_average = 0
        sync_iterations_average = 0
        disclosed_information_average = 0


    messages_sent_average = total_messages_sent / total_measurements
    traffic_volume_bits_average = total_traffic_volume_bits / total_measurements
    L = (2 ** Bits) // 2
    M = L * 0.25

    if verbose:
        print(f"K={K}, N={N}, b={Bits}, L={L}, M={M}")
        print(f"FER: {fer_average:.6f}")
        print(f"Среднее число фактических итераций синхронизации: {fact_sync_iterations_average:.6f}")
        print(f"Среднее число итераций: {sync_iterations_average:.6f}")
        print(f"Средняя раскрытая информация: {disclosed_information_average:.6f}")
        print(f"Среднее число отправленных сообщений: {messages_sent_average:.6f}")
        print(f"Средний объём трафика: {traffic_volume_bits_average:.6f} бит")

    return {
        "K": K, "N": N, "Bits": Bits, "L": L, "M": M,
        "fer_average": fer_average,
        "fact_sync_iterations_average": fact_sync_iterations_average,
        "sync_iterations_average": sync_iterations_average,
        "disclosed_information_average": disclosed_information_average,
        "messages_sent_average": messages_sent_average,
        "traffic_volume_bits_average": traffic_volume_bits_average,
        # "seconds": seconds,
    }




def select_parameters(frame_length):
    K = 3
    for Bits in range(8, 0, -1):
        denominator = K * Bits
        if frame_length % denominator == 0:
            N = frame_length // denominator
            if N >= 1:
                L = (2 ** Bits) // 2
                M = L * 0.25
                return K, N, Bits, L, M
    raise ValueError(
        "Для указанного размера кадра нельзя подобрать целые N и b "
        "при K=3 и 1 <= b <= 8. Введите другой размер кадра."
    )


def read_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt).strip())
            if value > 0:
                return value
            print("Значение должно быть положительным целым числом.")
        except ValueError:
            print("Введите целое число.")


def read_qber():
    while True:
        try:
            value = float(input("Введите желаемый QBER (например, 0.05 для 5%): ").strip().replace(',', '.'))
            if 0.0 <= value <= 1.0:
                return value
            print("QBER должен находиться в диапазоне от 0 до 1.")
        except ValueError:
            print("Введите числовое значение QBER.")


if __name__ == "__main__":
    print("Программа конфигурируемого согласования битовых последовательностей в системах квантовой коммуникации на основе автоматов Кантера-Кинцеля")
    print()

    qber = read_qber()

    while True:
        frame_length = read_positive_integer("Введите размер кадра F: ")
        try:
            K, N, Bits, L, M = select_parameters(frame_length)
            break
        except ValueError as error:
            print(error)

    measurements = read_positive_integer("Введите число репликаций: ")

    print("\nВыбранный набор параметров:")
    print(f"F={frame_length}, K={K}, N={N}, b={Bits}, L={L}, M={M}")
    print(f"QBER={qber:.6f} ({qber * 100:.2f}%)")
    print(f"Число репликаций: {measurements}")
    print(f"Ограничение числа итераций: {frame_length}")
    print("\nВыполняется синхронизация...")

    result = measure_single_parameter_set_parallel(
        K=K,
        N=N,
        Bits=Bits,
        kStart=1,
        kLimit=4,
        TCounterLimit=80,
        measurements=measurements,
        max_iterations=frame_length,
        Q=qber,
        verbose=False,
    )

    print("\nРезультаты эксперимента:")
    print(f"Параметры: K={K}, N={N}, b={Bits}, L={L}, M={M}, F={frame_length}")
    print(f"QBER: {qber:.6f} ({qber * 100:.2f}%)")
    print(f"Число репликаций: {measurements}")
    print(f"FER: {result['fer_average']:.6f}")
    print(f"Среднее число фактических итераций синхронизации: {result['fact_sync_iterations_average']:.6f}")
    print(f"Среднее число итераций: {result['sync_iterations_average']:.6f}")
    print(f"Средняя раскрытая информация: {result['disclosed_information_average']:.6f}")
    print(f"Среднее число отправленных сообщений: {result['messages_sent_average']:.6f}")
    print(f"Средний общий объём трафика: {result['traffic_volume_bits_average']:.6f} бит")

    result["QBER"] = qber
    result["frame_length_F"] = frame_length
    result["replications"] = measurements
    result["max_iterations"] = frame_length

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"TPM_results_F{frame_length}_{timestamp}.xlsx"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    pd.DataFrame([result]).to_excel(output_path, index=False, engine="openpyxl")
    print(f"\nРезультаты сохранены в Excel: {output_path}")
