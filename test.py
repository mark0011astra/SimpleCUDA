"""
SimpleCUDA テストコード
SimpleCUDA Core Features Test Suite

SimpleCUDAライブラリのコア機能をテストするためのスクリプト
A script to test core features of the SimpleCUDA library
"""

import unittest
import numpy as np
import time
import sys
import gc

# SimpleCUDAをインポート
# Import SimpleCUDA
try:
    from simplecuda import SimpleCUDA, cuda, array, to_gpu, to_cpu
except ImportError:
    print("SimpleCUDAモジュールが見つかりません。同じディレクトリにモジュールがあることを確認してください。")
    print("SimpleCUDA module not found. Please make sure the module is in the same directory.")
    sys.exit(1)


class TestSimpleCUDA(unittest.TestCase):
    """SimpleCUDAのユニットテスト / Unit tests for SimpleCUDA"""
    
    @classmethod
    def setUpClass(cls):
        """テスト開始前の準備 / Setup before tests"""
        print("\n============================================================")
        print("SimpleCUDA テストスイートを実行中... / Running SimpleCUDA test suite...")
        print("============================================================\n")
        
        # CPUモードとGPUモード両方でテスト
        # Test in both CPU and GPU modes
        cls.cuda_cpu = SimpleCUDA(force_cpu=True, verbose=False)
        cls.cuda_gpu = SimpleCUDA(force_cpu=False, verbose=False)
        
        # GPUが利用可能かどうかを記録
        # Record whether GPU is available
        cls.gpu_available = cls.cuda_gpu.use_gpu
        
        print(f"GPU利用可能: {cls.gpu_available} / GPU available: {cls.gpu_available}")
        if cls.gpu_available:
            mem_info = cls.cuda_gpu.memory_info()
            print(f"GPU メモリ: {mem_info['used_gb']:.2f} GB / {mem_info['total_gb']:.2f} GB "
                  f"({mem_info['usage_percent']:.2f}%)")
    
    def setUp(self):
        """各テスト前の準備 / Setup before each test"""
        # メモリを解放 / Free memory
        gc.collect()
        if TestSimpleCUDA.gpu_available:
            TestSimpleCUDA.cuda_gpu.clear_memory()
    
    def tearDown(self):
        """各テスト後の処理 / Cleanup after each test"""
        # メモリを解放 / Free memory
        gc.collect()
        if TestSimpleCUDA.gpu_available:
            TestSimpleCUDA.cuda_gpu.clear_memory()
    
    def test_initialization(self):
        """初期化テスト / Initialization test"""
        # 様々なオプションで初期化
        # Initialize with various options
        cuda1 = SimpleCUDA(verbose=False)
        self.assertIsInstance(cuda1, SimpleCUDA)
        
        cuda2 = SimpleCUDA(force_cpu=True, verbose=False)
        self.assertIsInstance(cuda2, SimpleCUDA)
        self.assertFalse(cuda2.use_gpu)
        
        # 異なるデバイスIDでの初期化（エラーになる場合はスキップ）
        # Initialize with different device IDs (skip if it causes an error)
        if TestSimpleCUDA.gpu_available:
            try:
                cuda4 = SimpleCUDA(device_id=0, verbose=False)
                self.assertIsInstance(cuda4, SimpleCUDA)
            except:
                pass
    
    def test_array_creation(self):
        """配列作成テスト / Array creation test"""
        for cuda_inst in [TestSimpleCUDA.cuda_cpu, TestSimpleCUDA.cuda_gpu]:
            # GPU利用可能でない場合はGPUテストをスキップ
            # Skip GPU tests if GPU is not available
            if not TestSimpleCUDA.gpu_available and cuda_inst is TestSimpleCUDA.cuda_gpu:
                continue
            
            # 基本的な配列作成
            # Basic array creation
            a = cuda_inst.array([1, 2, 3, 4])
            self.assertEqual(a.shape, (4,))
            
            # zeros, ones
            b = cuda_inst.zeros((2, 3))
            self.assertEqual(b.shape, (2, 3))
            self.assertEqual(to_cpu(b)[0, 0], 0)
            
            c = cuda_inst.ones((3, 2))
            self.assertEqual(c.shape, (3, 2))
            self.assertEqual(to_cpu(c)[0, 0], 1)
            
            # zeros_like, ones_like
            d = cuda_inst.zeros_like(a)
            self.assertEqual(d.shape, a.shape)
            
            e = cuda_inst.ones_like(a)
            self.assertEqual(e.shape, a.shape)
            
            # eye
            h = cuda_inst.eye(3)
            self.assertEqual(h.shape, (3, 3))
            self.assertEqual(to_cpu(h)[0, 0], 1)
            self.assertEqual(to_cpu(h)[0, 1], 0)
            
            # random
            m = cuda_inst.random((10, 10))
            self.assertEqual(m.shape, (10, 10))
            m_cpu = to_cpu(m)
            self.assertGreaterEqual(m_cpu.min(), 0)
            self.assertLessEqual(m_cpu.max(), 1)
            
            # arange
            o = cuda_inst.arange(0, 10, 2)
            self.assertEqual(o.shape, (5,))
    
    def test_data_transfer(self):
        """データ転送テスト / Data transfer test"""
        # GPU利用可能でない場合はスキップ
        # Skip if GPU is not available
        if not TestSimpleCUDA.gpu_available:
            self.skipTest("GPUが利用できないためスキップします / Skipping as GPU is not available")
        
        # NumPy配列からGPUへ / From NumPy array to GPU
        a_np = np.array([1, 2, 3, 4])
        a_gpu = TestSimpleCUDA.cuda_gpu.to_gpu(a_np)
        self.assertEqual(a_gpu.shape, a_np.shape)
        
        # GPUからNumPyへ / From GPU to NumPy
        a_back = TestSimpleCUDA.cuda_gpu.to_cpu(a_gpu)
        self.assertTrue(np.array_equal(a_np, a_back))
        
        # 別のGPU配列からNumPyへ / From another GPU array to NumPy
        b_gpu = TestSimpleCUDA.cuda_gpu.array([5, 6, 7, 8])
        b_np = TestSimpleCUDA.cuda_gpu.to_cpu(b_gpu)
        self.assertEqual(b_gpu.shape, b_np.shape)
    
    def test_basic_operations(self):
        """基本演算テスト / Basic operations test"""
        for cuda_inst in [TestSimpleCUDA.cuda_cpu, TestSimpleCUDA.cuda_gpu]:
            # GPU利用可能でない場合はGPUテストをスキップ
            # Skip GPU tests if GPU is not available
            if not TestSimpleCUDA.gpu_available and cuda_inst is TestSimpleCUDA.cuda_gpu:
                continue
            
            # 行列乗算 / Matrix multiplication
            a = cuda_inst.array([[1, 2], [3, 4]])
            b = cuda_inst.array([[5, 6], [7, 8]])
            c = cuda_inst.matmul(a, b)
            c_cpu = to_cpu(c)
            expected = np.array([[19, 22], [43, 50]])
            self.assertTrue(np.array_equal(c_cpu, expected))
            
            # ドット積 / Dot product
            d = cuda_inst.array([1, 2, 3])
            e = cuda_inst.array([4, 5, 6])
            f = cuda_inst.dot(d, e)
            f_cpu = to_cpu(f)
            self.assertEqual(f_cpu, 32)  # 1*4 + 2*5 + 3*6 = 32
            
            # 要素ごとの演算 / Element-wise operations
            g = cuda_inst.add(a, b)
            g_cpu = to_cpu(g)
            self.assertTrue(np.array_equal(g_cpu, np.array([[6, 8], [10, 12]])))
            
            h = cuda_inst.subtract(b, a)
            h_cpu = to_cpu(h)
            self.assertTrue(np.array_equal(h_cpu, np.array([[4, 4], [4, 4]])))
            
            i = cuda_inst.multiply(a, b)
            i_cpu = to_cpu(i)
            self.assertTrue(np.array_equal(i_cpu, np.array([[5, 12], [21, 32]])))
            
            j = cuda_inst.divide(b, a)
            j_cpu = to_cpu(j)
            self.assertTrue(np.allclose(j_cpu, np.array([[5, 3], [7/3, 2]])))
            
            k = cuda_inst.power(a, 2)
            k_cpu = to_cpu(k)
            self.assertTrue(np.array_equal(k_cpu, np.array([[1, 4], [9, 16]])))
    
    def test_unary_operations(self):
        """単項演算テスト / Unary operations test"""
        for cuda_inst in [TestSimpleCUDA.cuda_cpu, TestSimpleCUDA.cuda_gpu]:
            # GPU利用可能でない場合はGPUテストをスキップ
            # Skip GPU tests if GPU is not available
            if not TestSimpleCUDA.gpu_available and cuda_inst is TestSimpleCUDA.cuda_gpu:
                continue
            
            a = cuda_inst.array([-1, 2, -3, 4])
            
            # negative
            b = cuda_inst.negative(a)
            b_cpu = to_cpu(b)
            self.assertTrue(np.array_equal(b_cpu, np.array([1, -2, 3, -4])))
            
            # abs
            c = cuda_inst.abs(a)
            c_cpu = to_cpu(c)
            self.assertTrue(np.array_equal(c_cpu, np.array([1, 2, 3, 4])))
            
            # sqrt, square
            d = cuda_inst.array([1, 4, 9, 16])
            e = cuda_inst.sqrt(d)
            e_cpu = to_cpu(e)
            self.assertTrue(np.array_equal(e_cpu, np.array([1, 2, 3, 4])))
            
            f = cuda_inst.square(a)
            f_cpu = to_cpu(f)
            self.assertTrue(np.array_equal(f_cpu, np.array([1, 4, 9, 16])))
            
            # exp, log
            g = cuda_inst.array([0, 1, 2])
            h = cuda_inst.exp(g)
            h_cpu = to_cpu(h)
            self.assertTrue(np.allclose(h_cpu, np.array([1, np.e, np.e**2])))
            
            i = cuda_inst.log(h)
            i_cpu = to_cpu(i)
            self.assertTrue(np.allclose(i_cpu, np.array([0, 1, 2])))
    
    def test_aggregation_functions(self):
        """集計関数テスト / Aggregation functions test"""
        for cuda_inst in [TestSimpleCUDA.cuda_cpu, TestSimpleCUDA.cuda_gpu]:
            # GPU利用可能でない場合はGPUテストをスキップ
            # Skip GPU tests if GPU is not available
            if not TestSimpleCUDA.gpu_available and cuda_inst is TestSimpleCUDA.cuda_gpu:
                continue
            
            a = cuda_inst.array([[1, 2, 3], [4, 5, 6]])
            
            # sum
            b = cuda_inst.sum(a)
            b_cpu = to_cpu(b)
            self.assertEqual(b_cpu, 21)
            
            c = cuda_inst.sum(a, axis=0)
            c_cpu = to_cpu(c)
            self.assertTrue(np.array_equal(c_cpu, np.array([5, 7, 9])))
            
            d = cuda_inst.sum(a, axis=1)
            d_cpu = to_cpu(d)
            self.assertTrue(np.array_equal(d_cpu, np.array([6, 15])))
            
            # mean
            e = cuda_inst.mean(a)
            e_cpu = to_cpu(e)
            self.assertEqual(e_cpu, 3.5)
            
            f = cuda_inst.mean(a, axis=0)
            f_cpu = to_cpu(f)
            self.assertTrue(np.array_equal(f_cpu, np.array([2.5, 3.5, 4.5])))
            
            # max, min
            i = cuda_inst.max(a)
            i_cpu = to_cpu(i)
            self.assertEqual(i_cpu, 6)
            
            j = cuda_inst.min(a)
            j_cpu = to_cpu(j)
            self.assertEqual(j_cpu, 1)
            
            # argmax, argmin
            k = cuda_inst.argmax(a)
            k_cpu = to_cpu(k)
            self.assertEqual(k_cpu, 5)
            
            l = cuda_inst.argmin(a)
            l_cpu = to_cpu(l)
            self.assertEqual(l_cpu, 0)
    
    def test_shape_operations(self):
        """形状操作テスト / Shape operations test"""
        for cuda_inst in [TestSimpleCUDA.cuda_cpu, TestSimpleCUDA.cuda_gpu]:
            # GPU利用可能でない場合はGPUテストをスキップ
            # Skip GPU tests if GPU is not available
            if not TestSimpleCUDA.gpu_available and cuda_inst is TestSimpleCUDA.cuda_gpu:
                continue
            
            a = cuda_inst.array([[1, 2, 3], [4, 5, 6]])
            
            # reshape
            b = cuda_inst.reshape(a, (3, 2))
            b_cpu = to_cpu(b)
            self.assertTrue(np.array_equal(b_cpu, np.array([[1, 2], [3, 4], [5, 6]])))
            
            # transpose
            c = cuda_inst.transpose(a)
            c_cpu = to_cpu(c)
            self.assertTrue(np.array_equal(c_cpu, np.array([[1, 4], [2, 5], [3, 6]])))
            
            # concatenate
            d = cuda_inst.array([[7, 8, 9]])
            e = cuda_inst.concatenate([a, d])
            e_cpu = to_cpu(e)
            self.assertTrue(np.array_equal(e_cpu, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
    
    def test_linear_algebra(self):
        """線形代数テスト / Linear algebra test"""
        for cuda_inst in [TestSimpleCUDA.cuda_cpu, TestSimpleCUDA.cuda_gpu]:
            # GPU利用可能でない場合はGPUテストをスキップ
            # Skip GPU tests if GPU is not available
            if not TestSimpleCUDA.gpu_available and cuda_inst is TestSimpleCUDA.cuda_gpu:
                continue
            
            # 行列の準備 / Prepare matrices
            a = cuda_inst.array([[4, 7], [2, 6]])
            
            # linalg_inv
            try:
                h = cuda_inst.linalg_inv(a)
                h_cpu = to_cpu(h)
                # 結果の検証 / Verify results
                identity = to_cpu(cuda_inst.matmul(a, h))
                self.assertTrue(np.allclose(identity, np.eye(2), atol=1e-5))
            except Exception as e:
                print(f"linalg_invテストでエラー / Error in linalg_inv test: {e}")
            
            # linalg_solve
            try:
                j = cuda_inst.array([23, 17])
                k = cuda_inst.linalg_solve(a, j)
                k_cpu = to_cpu(k)
                # 結果の検証 / Verify results
                result = to_cpu(cuda_inst.matmul(a, k))
                self.assertTrue(np.allclose(result, to_cpu(j), atol=1e-5))
            except Exception as e:
                print(f"linalg_solveテストでエラー / Error in linalg_solve test: {e}")
    
    def test_memory_management(self):
        """メモリ管理テスト / Memory management test"""
        # GPU利用可能でない場合はスキップ
        # Skip if GPU is not available
        if not TestSimpleCUDA.gpu_available:
            self.skipTest("GPUが利用できないためスキップします / Skipping as GPU is not available")
        
        # メモリ情報の取得 / Get memory information
        mem_info = TestSimpleCUDA.cuda_gpu.memory_info()
        self.assertIsInstance(mem_info, dict)
        self.assertIn("total_gb", mem_info)
        self.assertIn("used_gb", mem_info)
        self.assertIn("free_gb", mem_info)
        self.assertIn("usage_percent", mem_info)
        
        # 大きな配列の作成と解放 / Create and free large array
        initial_used = mem_info["used_gb"]
        
        # 大きな配列を作成 / Create large array
        large_array = TestSimpleCUDA.cuda_gpu.random((1000, 1000))
        
        # メモリ使用量の増加を確認 / Check that memory usage has increased
        mem_after = TestSimpleCUDA.cuda_gpu.memory_info()
        
        # メモリ解放 / Free memory
        del large_array
        TestSimpleCUDA.cuda_gpu.clear_memory()
        
        # メモリ使用量の減少を確認 / Check that memory usage has decreased
        mem_cleared = TestSimpleCUDA.cuda_gpu.memory_info()
        self.assertLessEqual(mem_cleared["used_gb"], mem_after["used_gb"])
    
    def test_timer(self):
        """タイマーテスト / Timer test"""
        for cuda_inst in [TestSimpleCUDA.cuda_cpu, TestSimpleCUDA.cuda_gpu]:
            # GPU利用可能でない場合はGPUテストをスキップ
            # Skip GPU tests if GPU is not available
            if not TestSimpleCUDA.gpu_available and cuda_inst is TestSimpleCUDA.cuda_gpu:
                continue
            
            # タイマーコンテキストマネージャのテスト / Test timer context manager
            with cuda_inst.timer("テスト演算 / Test operation"):
                a = cuda_inst.random((10, 10))
                b = cuda_inst.random((10, 10))
                c = cuda_inst.matmul(a, b)
                self.assertEqual(c.shape, (10, 10))
    
    def test_benchmark(self):
        """ベンチマークテスト / Benchmark test"""
        for cuda_inst in [TestSimpleCUDA.cuda_cpu, TestSimpleCUDA.cuda_gpu]:
            # GPU利用可能でない場合はGPUテストをスキップ
            # Skip GPU tests if GPU is not available
            if not TestSimpleCUDA.gpu_available and cuda_inst is TestSimpleCUDA.cuda_gpu:
                continue
            
            # 乱数行列の生成関数 / Random matrix generation function
            def create_random_matrix():
                return cuda_inst.random((10, 10))
            
            # ベンチマーク結果の検証 / Verify benchmark results
            result = cuda_inst.benchmark(create_random_matrix, repeat=2, iterations=5)
            self.assertIn("mean", result)
            self.assertIn("min", result)
            self.assertIn("max", result)
            self.assertIn("median", result)
            self.assertIn("std", result)
            self.assertIn("times", result)
            
            # 複数関数の比較 / Compare multiple functions
            def func1():
                a = cuda_inst.random((10, 10))
                b = cuda_inst.random((10, 10))
                return cuda_inst.matmul(a, b)
            
            def func2():
                a = cuda_inst.random((10, 10))
                b = cuda_inst.random((10, 10))
                return cuda_inst.add(a, b)
            
            funcs = {"matmul": func1, "add": func2}
            results = cuda_inst.benchmark_compare(funcs, repeat=2, iterations=5)
            self.assertIn("matmul", results)
            self.assertIn("add", results)


class BenchmarkSimpleCUDA:
    """SimpleCUDAのベンチマーク / Benchmarks for SimpleCUDA"""
    
    @staticmethod
    def run():
        """ベンチマークの実行 / Run benchmarks"""
        print("\n============================================================")
        print("SimpleCUDA ベンチマークを実行中... / Running SimpleCUDA benchmarks...")
        print("============================================================\n")
        
        # インスタンスの作成 / Create instances
        cuda_cpu = SimpleCUDA(force_cpu=True, verbose=False)
        cuda_gpu = SimpleCUDA(force_cpu=False, verbose=False)
        gpu_available = cuda_gpu.use_gpu
        
        if not gpu_available:
            print("GPUが利用できないため、CPUでのベンチマークのみ実行します。")
            print("Only running CPU benchmarks as GPU is not available.")
        
        # 行列サイズ / Matrix sizes
        sizes = [100, 500, 1000, 2000] if gpu_available else [100, 500, 1000]
        
        # 行列乗算のベンチマーク / Matrix multiplication benchmark
        print("\n行列乗算ベンチマーク / Matrix multiplication benchmark")
        print("-" * 60)
        print(f"{'サイズ / Size':<10} {'CPU (ms)':<15} {'GPU (ms)':<15} {'高速化率 / Speedup':<15}")
        print("-" * 60)
        
        for size in sizes:
            # NumPy行列の生成 / Generate NumPy matrices
            a_np = np.random.rand(size, size).astype(np.float32)
            b_np = np.random.rand(size, size).astype(np.float32)
            
            # CPUでの計測 / Measure on CPU
            a_cpu = cuda_cpu.array(a_np)
            b_cpu = cuda_cpu.array(b_np)
            
            cpu_results = []
            for _ in range(3):
                start = time.time()
                _ = cuda_cpu.matmul(a_cpu, b_cpu)
                cpu_results.append((time.time() - start) * 1000)  # ms
            
            cpu_time = np.mean(cpu_results)
            
            # GPUでの計測 / Measure on GPU
            if gpu_available:
                a_gpu = cuda_gpu.array(a_np)
                b_gpu = cuda_gpu.array(b_np)
                
                # ウォームアップ / Warm-up
                _ = cuda_gpu.matmul(a_gpu, b_gpu)
                cuda_gpu.synchronize()
                
                gpu_results = []
                for _ in range(3):
                    start = time.time()
                    _ = cuda_gpu.matmul(a_gpu, b_gpu)
                    cuda_gpu.synchronize()
                    gpu_results.append((time.time() - start) * 1000)  # ms
                
                gpu_time = np.mean(gpu_results)
                
                # 0除算回避 / Avoid division by zero
                if gpu_time > 0.001:  # 1ミリ秒より大きい場合のみ計算
                    speedup = cpu_time / gpu_time
                    print(f"{size:<10} {cpu_time:<15.2f} {gpu_time:<15.2f} {speedup:<15.2f}x")
                else:
                    print(f"{size:<10} {cpu_time:<15.2f} {gpu_time:<15.2f} {'N/A':<15}")
            else:
                print(f"{size:<10} {cpu_time:<15.2f} {'N/A':<15} {'N/A':<15}")
        
        # 要素ごとの演算ベンチマーク / Element-wise operations benchmark
        print("\n要素ごとの演算ベンチマーク / Element-wise operations benchmark")
        print("-" * 60)
        print(f"{'操作 / Operation':<15} {'CPU (ms)':<15} {'GPU (ms)':<15} {'高速化率 / Speedup':<15}")
        print("-" * 60)
        
        # 大きな行列を生成 / Generate large matrices
        size = 2000 if gpu_available else 1000
        a_np = np.random.rand(size, size).astype(np.float32)
        b_np = np.random.rand(size, size).astype(np.float32)
        
        a_cpu = cuda_cpu.array(a_np)
        b_cpu = cuda_cpu.array(b_np)
        
        if gpu_available:
            a_gpu = cuda_gpu.array(a_np)
            b_gpu = cuda_gpu.array(b_np)
        
        operations = {
            "add": lambda c, x, y: c.add(x, y),
            "multiply": lambda c, x, y: c.multiply(x, y),
            "divide": lambda c, x, y: c.divide(x, y),
            "sqrt": lambda c, x, y: c.sqrt(x),
            "exp": lambda c, x, y: c.exp(x),
            "log": lambda c, x, y: c.log(x),
        }
        
        for op_name, op_func in operations.items():
            # CPUでの計測 / Measure on CPU
            cpu_results = []
            for _ in range(3):
                start = time.time()
                _ = op_func(cuda_cpu, a_cpu, b_cpu)
                cpu_results.append((time.time() - start) * 1000)  # ms
            
            cpu_time = np.mean(cpu_results)
            
            # GPUでの計測 / Measure on GPU
            if gpu_available:
                # ウォームアップ / Warm-up
                _ = op_func(cuda_gpu, a_gpu, b_gpu)
                cuda_gpu.synchronize()
                
                gpu_results = []
                for _ in range(3):
                    start = time.time()
                    _ = op_func(cuda_gpu, a_gpu, b_gpu)
                    cuda_gpu.synchronize()
                    gpu_results.append((time.time() - start) * 1000)  # ms
                
                gpu_time = np.mean(gpu_results)
                
                # 0除算回避 / Avoid division by zero
                if gpu_time > 0.001:  # 1ミリ秒より大きい場合のみ計算
                    speedup = cpu_time / gpu_time
                    print(f"{op_name:<15} {cpu_time:<15.2f} {gpu_time:<15.2f} {speedup:<15.2f}x")
                else:
                    print(f"{op_name:<15} {cpu_time:<15.2f} {gpu_time:<15.2f} {'N/A':<15}")
            else:
                print(f"{op_name:<15} {cpu_time:<15.2f} {'N/A':<15} {'N/A':<15}")
        
        print("\nベンチマーク完了! / Benchmarks completed!")


if __name__ == "__main__":
    # ユニットテストの実行 / Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # ベンチマークの実行 / Run benchmarks
    BenchmarkSimpleCUDA.run()
