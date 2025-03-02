"""
SimpleCUDA: シンプルで高速なPython CUDA ラッパー
SimpleCUDA: Simple and Fast Python CUDA Wrapper (Core Features)

NumPyライクなインターフェースでGPU計算を簡単に行える軽量ライブラリ
A lightweight library that makes GPU computing easy with NumPy-like interface
"""

import numpy as np
import warnings
import time
import functools
import contextlib
import gc
import platform
import os

# CUDAの利用可能性を確認 / Check CUDA availability
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn("CuPyがインストールされていません。CPUモードで実行します。\n"
                  "CuPy is not installed. Running in CPU mode.")
except Exception as e:
    CUDA_AVAILABLE = False
    warnings.warn(f"CUDAの初期化中にエラーが発生しました: {e}\n"
                  f"Error occurred during CUDA initialization: {e}")


def _ensure_array(func):
    """
    配列形式を保証するデコレータ
    Decorator to ensure array format
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 引数を配列に変換 / Convert arguments to arrays
        new_args = []
        for arg in args:
            if arg is not None and not (hasattr(arg, 'shape') or hasattr(arg, 'size')):
                try:
                    arg = self.xp.asarray(arg)
                except:
                    pass
            new_args.append(arg)
        
        # 関数を実行 / Execute function
        result = func(self, *new_args, **kwargs)
        return result
    
    return wrapper


class SimpleCUDA:
    """シンプルなCUDAラッパー / Simple CUDA wrapper"""
    
    def __init__(self, force_cpu=False, device_id=0, verbose=False, autosync=True):
        """
        SimpleCUDAの初期化 / Initialize SimpleCUDA
        
        Args:
            force_cpu: TrueにするとCPUモードを強制 / Force CPU mode if True
            device_id: 使用するGPUデバイスID / GPU device ID to use
            verbose: 詳細情報を表示するか / Whether to display detailed information
            autosync: 演算後に自動的に同期するか / Whether to automatically synchronize after operations
        """
        self.verbose = verbose
        self.autosync = autosync
        
        if verbose:
            print(f"SimpleCUDA初期化... / SimpleCUDA initialization...")
        
        # デフォルトのデータ型を設定 / Set default data type
        self.float_dtype = np.float32
        
        # GPUモードが使用可能か確認 / Check if GPU mode is available
        self.use_gpu = CUDA_AVAILABLE and not force_cpu
        
        if self.use_gpu:
            try:
                # デバイスの設定 / Set device
                self.device = cp.cuda.Device(device_id)
                self.device.use()
                self.xp = cp
                
                # ウォームアップ（初回実行の遅延を防ぐ） / Warm-up (prevent delay in first execution)
                a = self.xp.array([1, 2, 3], dtype=self.float_dtype)
                b = self.xp.array([4, 5, 6], dtype=self.float_dtype)
                c = self.xp.dot(a, b)
                self.synchronize()
                
                if verbose:
                    device_props = cp.cuda.runtime.getDeviceProperties(device_id)
                    device_name = device_props["name"].decode("utf-8")
                    print(f"GPU初期化完了: {device_name} / GPU initialization complete: {device_name}")
                    
                    # GPUメモリ情報 / GPU memory information
                    mem_info = self.memory_info()
                    print(f"GPU メモリ / GPU memory: {mem_info['used_gb']:.2f} GB / {mem_info['total_gb']:.2f} GB "
                          f"({mem_info['usage_percent']:.2f}%)")
            
            except Exception as e:
                if verbose:
                    print(f"GPU初期化エラー: {e} / GPU initialization error: {e}")
                    print(f"CPUモードにフォールバック / Falling back to CPU mode")
                self.use_gpu = False
                self.xp = np
        else:
            self.xp = np
            if verbose:
                print("CPUモードで実行中 / Running in CPU mode")
    
    def synchronize(self):
        """
        GPU操作を同期 / Synchronize GPU operations
        """
        if self.use_gpu:
            self.device.synchronize()
    
    # 配列作成 / Array creation
    @_ensure_array
    def array(self, data, dtype=None):
        """配列を作成 / Create array"""
        if dtype is None:
            if isinstance(data, (list, tuple)) and all(isinstance(x, (int, float)) for x in data):
                dtype = self.float_dtype
        return self.xp.array(data, dtype=dtype)
    
    def zeros(self, shape, dtype=None):
        """ゼロ配列を作成 / Create zeros array"""
        if dtype is None:
            dtype = self.float_dtype
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        """1で埋められた配列を作成 / Create ones array"""
        if dtype is None:
            dtype = self.float_dtype
        return self.xp.ones(shape, dtype=dtype)
    
    def zeros_like(self, a, dtype=None):
        """入力と同じ形状のゼロ配列を作成 / Create zeros array with the same shape"""
        return self.xp.zeros_like(a, dtype=dtype)
    
    def ones_like(self, a, dtype=None):
        """入力と同じ形状の1で埋められた配列を作成 / Create ones array with the same shape"""
        return self.xp.ones_like(a, dtype=dtype)
    
    def eye(self, n, dtype=None):
        """単位行列を作成 / Create identity matrix"""
        if dtype is None:
            dtype = self.float_dtype
        return self.xp.eye(n, dtype=dtype)
    
    def random(self, shape, dtype=None):
        """ランダム配列を作成 / Create random array"""
        if dtype is None:
            dtype = self.float_dtype
        return self.xp.random.random(shape).astype(dtype)
    
    def arange(self, start, stop=None, step=1, dtype=None):
        """連続した値の配列を作成 / Create array with evenly spaced values"""
        return self.xp.arange(start, stop, step, dtype=dtype)
    
    # データ転送 / Data transfer
    def to_gpu(self, array):
        """NumPy配列をGPUに転送 / Transfer NumPy array to GPU"""
        if self.use_gpu:
            if isinstance(array, np.ndarray):
                return self.xp.asarray(array)
            else:
                try:
                    return self.xp.asarray(array)
                except:
                    return array
        return array
    
    def to_cpu(self, array):
        """配列をCPUに転送 / Transfer array to CPU"""
        if self.use_gpu and hasattr(array, "get"):
            return array.get()
        return array
    
    # 基本演算 / Basic operations
    @_ensure_array
    def matmul(self, a, b):
        """行列乗算 / Matrix multiplication"""
        result = self.xp.matmul(a, b)
        if self.use_gpu and self.autosync:
            self.synchronize()
        return result
    
    @_ensure_array
    def dot(self, a, b):
        """内積 / Dot product"""
        result = self.xp.dot(a, b)
        if self.use_gpu and self.autosync:
            self.synchronize()
        return result
    
    @_ensure_array
    def add(self, a, b):
        """要素ごとの加算 / Element-wise addition"""
        return self.xp.add(a, b)
    
    @_ensure_array
    def subtract(self, a, b):
        """要素ごとの減算 / Element-wise subtraction"""
        return self.xp.subtract(a, b)
    
    @_ensure_array
    def multiply(self, a, b):
        """要素ごとの乗算 / Element-wise multiplication"""
        return self.xp.multiply(a, b)
    
    @_ensure_array
    def divide(self, a, b):
        """要素ごとの除算 / Element-wise division"""
        return self.xp.divide(a, b)
    
    @_ensure_array
    def power(self, a, b):
        """要素ごとの累乗 / Element-wise power"""
        return self.xp.power(a, b)
    
    # 単項演算 / Unary operations
    @_ensure_array
    def negative(self, a):
        """符号反転 / Negative"""
        return self.xp.negative(a)
    
    @_ensure_array
    def abs(self, a):
        """絶対値 / Absolute value"""
        return self.xp.abs(a)
    
    @_ensure_array
    def sqrt(self, a):
        """平方根 / Square root"""
        return self.xp.sqrt(a)
    
    @_ensure_array
    def square(self, a):
        """2乗 / Square"""
        return self.xp.square(a)
    
    @_ensure_array
    def exp(self, a):
        """指数関数 / Exponential"""
        return self.xp.exp(a)
    
    @_ensure_array
    def log(self, a):
        """自然対数 / Natural logarithm"""
        return self.xp.log(a)
    
    # 集計関数 / Aggregation functions
    @_ensure_array
    def sum(self, a, axis=None, keepdims=False):
        """合計 / Sum"""
        return self.xp.sum(a, axis=axis, keepdims=keepdims)
    
    @_ensure_array
    def mean(self, a, axis=None, keepdims=False):
        """平均 / Mean"""
        return self.xp.mean(a, axis=axis, keepdims=keepdims)
    
    @_ensure_array
    def max(self, a, axis=None, keepdims=False):
        """最大値 / Maximum value"""
        return self.xp.max(a, axis=axis, keepdims=keepdims)
    
    @_ensure_array
    def min(self, a, axis=None, keepdims=False):
        """最小値 / Minimum value"""
        return self.xp.min(a, axis=axis, keepdims=keepdims)
    
    @_ensure_array
    def argmax(self, a, axis=None):
        """最大値のインデックス / Index of maximum value"""
        return self.xp.argmax(a, axis=axis)
    
    @_ensure_array
    def argmin(self, a, axis=None):
        """最小値のインデックス / Index of minimum value"""
        return self.xp.argmin(a, axis=axis)
    
    # 形状操作 / Shape operations
    def reshape(self, a, newshape):
        """形状変更 / Reshape"""
        return self.xp.reshape(a, newshape)
    
    def transpose(self, a, axes=None):
        """転置 / Transpose"""
        return self.xp.transpose(a, axes=axes)
    
    def concatenate(self, arrays, axis=0):
        """配列の連結 / Concatenate arrays"""
        return self.xp.concatenate(arrays, axis=axis)
    
    # 行列の分解と演算 / Matrix decomposition and operations
    @_ensure_array
    def linalg_inv(self, a):
        """逆行列 / Inverse matrix"""
        if self.use_gpu:
            try:
                result = self.xp.linalg.inv(a)
            except Exception as e:
                if self.verbose:
                    print(f"GPUでの逆行列計算に失敗しました: {e} / Failed to compute inverse matrix on GPU: {e}")
                # NumPyの実装にフォールバック / Fallback to NumPy implementation
                a_cpu = self.to_cpu(a)
                result = np.linalg.inv(a_cpu)
                result = self.to_gpu(result)
        else:
            result = self.xp.linalg.inv(a)
        
        if self.use_gpu and self.autosync:
            self.synchronize()
        
        return result
    
    @_ensure_array
    def linalg_solve(self, a, b):
        """線形方程式を解く / Solve linear equations"""
        if self.use_gpu:
            try:
                result = self.xp.linalg.solve(a, b)
            except Exception as e:
                if self.verbose:
                    print(f"GPUでの線形方程式計算に失敗しました: {e} / Failed to solve linear equations on GPU: {e}")
                # NumPyの実装にフォールバック / Fallback to NumPy implementation
                a_cpu = self.to_cpu(a)
                b_cpu = self.to_cpu(b)
                result = np.linalg.solve(a_cpu, b_cpu)
                result = self.to_gpu(result)
        else:
            result = self.xp.linalg.solve(a, b)
        
        if self.use_gpu and self.autosync:
            self.synchronize()
        
        return result
    
    # メモリ管理 / Memory management
    def memory_info(self):
        """GPUメモリ情報を取得 / Get GPU memory information"""
        if not self.use_gpu:
            return {"message": "CPUモードで実行中 / Running in CPU mode"}
        
        free, total = self.device.mem_info
        used = total - free
        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "usage_percent": (used / total) * 100
        }
    
    def clear_memory(self):
        """不要なメモリを解放 / Free unnecessary memory"""
        if self.use_gpu:
            self.synchronize()
            try:
                cp.get_default_memory_pool().free_all_blocks()
                if hasattr(cp.cuda, 'get_default_pinned_memory_pool'):
                    cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                if self.verbose:
                    print(f"メモリ解放中にエラーが発生しました: {e} / Error occurred during memory release: {e}")
        
        # CPUメモリの明示的なGC / Explicit GC for CPU memory
        gc.collect()
    
    # ユーティリティ / Utilities
    @contextlib.contextmanager
    def timer(self, name="Operation", sync=True):
        """
        処理時間を測定するコンテキストマネージャ
        Context manager for measuring processing time
        
        with cuda.timer("MatMul"):
            c = cuda.matmul(a, b)
        """
        if sync and self.use_gpu:
            self.synchronize()
        
        start = time.time()
        yield
        
        if sync and self.use_gpu:
            self.synchronize()
        
        end = time.time()
        elapsed = end - start
        
        if self.verbose:
            print(f"{name}: {elapsed * 1000:.2f} ms")
        
        return elapsed
    
    def info(self):
        """
        システム情報を表示
        Display system information
        """
        print(f"===== SimpleCUDA 情報 / SimpleCUDA Information =====")
        print(f"モード / Mode: {'GPU' if self.use_gpu else 'CPU'}")
        
        if self.use_gpu:
            device_props = cp.cuda.runtime.getDeviceProperties(self.device.id)
            device_name = device_props["name"].decode("utf-8")
            print(f"GPU: {device_name}")
            print(f"CUDA バージョン / CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
            print(f"CuPy バージョン / CuPy version: {cp.__version__}")
            
            mem_info = self.memory_info()
            print(f"GPU メモリ / GPU memory: {mem_info['used_gb']:.2f} GB / {mem_info['total_gb']:.2f} GB "
                  f"({mem_info['usage_percent']:.2f}%)")
        else:
            print(f"NumPy バージョン / NumPy version: {np.__version__}")
        
        print(f"Python バージョン / Python version: {platform.python_version()}")
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"自動同期モード / Auto synchronization mode: {'有効 / Enabled' if self.autosync else '無効 / Disabled'}")
        print(f"================================================")
    
    # ベンチマーク / Benchmark
    def benchmark(self, func, *args, repeat=3, iterations=10, sync=True, **kwargs):
        """
        関数の実行時間をベンチマーク
        Benchmark execution time of a function
        
        Args:
            func: ベンチマークする関数 / Function to benchmark
            args: 関数に渡す引数 / Arguments to pass to the function
            repeat: 反復回数（各計測につき） / Number of repetitions (per measurement)
            iterations: 関数内でのループ回数 / Number of iterations within the function
            sync: GPU同期を行うか / Whether to synchronize GPU
            kwargs: 関数に渡すキーワード引数 / Keyword arguments to pass to the function
        """
        # ウォームアップ / Warm-up
        for _ in range(3):
            func(*args, **kwargs)
        
        if self.use_gpu and sync:
            self.synchronize()
        
        # 計測 / Measurement
        times = []
        for r in range(repeat):
            if self.verbose and repeat > 1:
                print(f"ベンチマーク実行 {r+1}/{repeat} / Benchmark run {r+1}/{repeat}")
            
            start = time.time()
            
            # 複数回実行して平均時間を取得 / Execute multiple times to get average time
            for _ in range(iterations):
                func(*args, **kwargs)
                if self.use_gpu and sync:
                    self.synchronize()
            
            end = time.time()
            # イテレーションあたりの時間を計算 / Calculate time per iteration
            times.append((end - start) / iterations)
        
        result = {
            "mean": np.mean(times),
            "min": np.min(times),
            "max": np.max(times),
            "median": np.median(times),
            "std": np.std(times),
            "times": times
        }
        
        if self.verbose:
            print(f"ベンチマーク結果 / Benchmark results:")
            print(f"  平均時間 / Mean time: {result['mean'] * 1000:.4f} ms")
            print(f"  最小時間 / Min time: {result['min'] * 1000:.4f} ms")
            print(f"  最大時間 / Max time: {result['max'] * 1000:.4f} ms")
            print(f"  標準偏差 / Std dev: {result['std'] * 1000:.4f} ms")
        
        return result
    
    def benchmark_compare(self, funcs_dict, *args, repeat=3, iterations=10, **kwargs):
        """
        複数の関数の実行時間を比較
        Compare execution time of multiple functions
        
        Args:
            funcs_dict: 関数の辞書 {名前: 関数} / Dictionary of functions {name: function}
            args: 関数に渡す引数 / Arguments to pass to the function
            repeat: 反復回数（各計測につき） / Number of repetitions (per measurement)
            iterations: 関数内でのループ回数 / Number of iterations within the function
            kwargs: 関数に渡すキーワード引数 / Keyword arguments to pass to the function
        """
        results = {}
        baseline = None
        
        for name, func in funcs_dict.items():
            if self.verbose:
                print(f"\nベンチマーク: {name} / Benchmark: {name}")
            
            result = self.benchmark(func, *args, repeat=repeat, iterations=iterations, **kwargs)
            results[name] = result
            
            if baseline is None:
                baseline = result["mean"]
        
        if self.verbose:
            print("\n比較結果 / Comparison results:")
            print(f"{'名前 / Name':<20} {'時間 / Time (ms)':<15} {'相対速度 / Relative speed':<20}")
            print(f"{'-' * 55}")
            
            # 平均時間で並べ替え / Sort by mean time
            sorted_results = sorted(results.items(), key=lambda x: x[1]["mean"])
            
            for name, result in sorted_results:
                # ゼロ除算を回避 / Avoid division by zero
                if result["mean"] > 0 and baseline > 0:
                    relative = baseline / result["mean"]
                    print(f"{name:<20} {result['mean'] * 1000:>15.4f} {relative:>20.2f}x")
                else:
                    print(f"{name:<20} {result['mean'] * 1000:>15.4f} {'N/A':>20}")
        
        return results


# グローバルインスタンス作成 / Create global instance
cuda = SimpleCUDA()

# 便利なエイリアス / Convenient aliases
array = cuda.array
zeros = cuda.zeros
ones = cuda.ones
zeros_like = cuda.zeros_like
ones_like = cuda.ones_like
to_gpu = cuda.to_gpu
to_cpu = cuda.to_cpu
matmul = cuda.matmul
dot = cuda.dot

# デモ関数 / Demo function
def demo():
    """SimpleCUDAのデモを実行 / Run SimpleCUDA demo"""
    print("\nSimpleCUDA デモ / Demo")
    print("=================")
    
    # GPU情報表示 / Display GPU information
    if cuda.use_gpu:
        cuda.info()
    else:
        print("CPUモードで実行中 / Running in CPU mode")
    
    # 配列演算テスト / Array operation test
    print("\n配列演算テスト / Array operation test:")
    with cuda.timer("配列作成とGPU転送 / Array creation and GPU transfer"):
        a = array([[1, 2], [3, 4]])
        b = array([[5, 6], [7, 8]])
    
    with cuda.timer("行列乗算 / Matrix multiplication"):
        c = matmul(a, b)
    
    print(f"  行列乗算結果 / Matrix multiplication result:\n{to_cpu(c)}")
    
    # 線形代数テスト / Linear algebra test
    print("\n線形代数テスト / Linear algebra test:")
    a_inv = array([[4, 7], [2, 6]])
    
    with cuda.timer("逆行列計算 / Inverse matrix calculation"):
        a_inv_result = cuda.linalg_inv(a_inv)
    
    print(f"  元の行列 / Original matrix:\n{to_cpu(a_inv)}")
    print(f"  逆行列 / Inverse matrix:\n{to_cpu(a_inv_result)}")
    print(f"  検証 / Verification:\n{to_cpu(matmul(a_inv, a_inv_result))}")
    
    # サイズの異なる行列での速度比較 / Speed comparison with different matrix sizes
    sizes = [1000, 2000, 3000] if cuda.use_gpu else [500, 1000, 1500]
    
    print("\n行列サイズごとの速度比較 / Speed comparison by matrix size:")
    for size in sizes:
        print(f"\n  {size}x{size}行列の乗算 / Multiplication of {size}x{size} matrices")
        
        # NumPyでのテスト / Test with NumPy
        a_np = np.random.rand(size, size).astype(np.float32)
        b_np = np.random.rand(size, size).astype(np.float32)
        
        # SimpleCUDAでの計算時間 / Computation time with SimpleCUDA
        # ここでは測定のために一度GPUに転送 / Transfer to GPU once for measurement
        a_sc = array(a_np)
        b_sc = array(b_np)
        
        # ウォームアップ / Warm-up
        _ = matmul(a_sc, b_sc)
        if cuda.use_gpu:
            cuda.synchronize()
        
        # 複数回実行して平均を取る / Execute multiple times and take the average
        iterations = 5
        
        # NumPyでの計算時間 / Computation time with NumPy
        np_times = []
        for _ in range(iterations):
            start = time.time()
            _ = np.matmul(a_np, b_np)
            np_times.append(time.time() - start)
        np_time = np.mean(np_times)
        print(f"    NumPy時間 / NumPy time: {np_time:.6f}秒 / seconds")
        
        # SimpleCUDAでの計算時間 / Computation time with SimpleCUDA
        sc_times = []
        for _ in range(iterations):
            start = time.time()
            _ = matmul(a_sc, b_sc)
            if cuda.use_gpu:
                cuda.synchronize()
            sc_times.append(time.time() - start)
        sc_time = np.mean(sc_times)
        print(f"    SimpleCUDA時間 / SimpleCUDA time: {sc_time:.6f}秒 / seconds")
        
        # 高速化率 / Speedup ratio
        if sc_time > 0 and np_time > 0:
            speedup = np_time / sc_time
            print(f"    高速化率 / Speedup ratio: {speedup:.2f}倍 / times")
        else:
            print("    高速化率: 計算できません / Speedup ratio: Cannot calculate")
    
    print("\nデモ完了! / Demo completed!")

if __name__ == "__main__":
    demo()
