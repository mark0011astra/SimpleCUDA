![simpleCUDA](img01.png)

# SimpleCUDA
NumPyのようなインターフェースで簡単にCUDAを使用したGPU演算を行えるライブラリ。A library that allows users to easily perform GPU operations using CUDA with a NumPy-like interface

SimpleCUDAは、NumPyと同様のインターフェースを保ちながらGPUの計算能力を活用できる、シンプルかつ強力なPythonライブラリです。科学計算、機械学習の前処理、データ解析など、計算処理を必要とするあらゆる場面で、コードの変更を最小限に抑えつつGPU高速化を実現します。

### 特徴

- **直感的なAPI**: NumPyと同様のインターフェースで、学習コストを最小化
- **高速処理**: CPUと比較して最大40倍の高速化を実現
- **コンパクト設計**: 必要最小限の機能に集中した軽量実装
- **自動フォールバック**: GPU非搭載環境でも自動的にCPUで動作
- **強力なエラーハンドリング**: 開発環境の違いによる問題を最小化
- **日本語/英語ドキュメント**: 豊富なコメントとドキュメントを完備

## 必要要件

- Python 3.7以上
- NumPy
- CuPy（GPU機能を使用する場合）
- CUDA対応NVIDIA GPU（GPU機能を使用する場合）

## インストール方法

1. リポジトリをクローン:

```bash
git clone https://github.com/yourusername/simplecuda.git
```

2. Pythonコードと同じディレクトリに`simplecuda.py`を配置するか、Pythonのパスが通っている場所に配置します。

3. そのまま`import simplecuda`でライブラリを使用できます:

```python
from simplecuda import cuda, array, to_gpu, to_cpu
```

## 基本的な使い方

```python
import numpy as np
from simplecuda import cuda, array, matmul, to_cpu

# NumPy配列の作成
a_np = np.random.rand(1000, 1000).astype(np.float32)
b_np = np.random.rand(1000, 1000).astype(np.float32)

# GPU配列への変換
a = array(a_np)  # 自動的にGPUに転送されます
b = array(b_np)

# GPU上での行列乗算
c = matmul(a, b)

# 結果をCPUに戻す
c_np = to_cpu(c)

print(f"GPU計算結果の形状: {c_np.shape}")
```

## サポートしている主な機能

### 配列作成

```python
# 様々な方法での配列作成
a = array([1, 2, 3, 4])
b = zeros((3, 4))
c = ones((2, 5))
d = eye(3)  # 3x3の単位行列
e = random((5, 5))  # 0-1の一様乱数
```

### 基本演算

```python
# 基本的な行列・配列演算
c = matmul(a, b)  # 行列乗算
d = dot(a, b)     # ドット積
e = add(a, b)     # 要素ごとの加算
f = subtract(a, b)  # 要素ごとの減算
g = multiply(a, b)  # 要素ごとの乗算
h = divide(a, b)    # 要素ごとの除算
```

### その他の数学関数

```python
# 単項演算と数学関数
a_sqrt = sqrt(a)    # 平方根
a_exp = exp(a)      # 指数関数
a_log = log(a)      # 自然対数
a_abs = abs(a)      # 絶対値
```

### 集計関数

```python
# データの集計
total = sum(a)         # 合計
avg = mean(a)          # 平均
max_val = max(a)       # 最大値
min_val = min(a)       # 最小値
max_idx = argmax(a)    # 最大値のインデックス
```

### 形状操作

```python
# 形状の変更
a_reshaped = reshape(a, (2, 5))    # 形状変更
a_t = transpose(a)                 # 転置
```

### パフォーマンス測定

```python
# 計算時間の測定
with cuda.timer("行列乗算"):
    c = matmul(a, b)
```

## ベンチマーク結果

以下は2000x2000行列での演算のベンチマーク結果です（NVIDIAのRTX3070使用時）:

| 演算         | CPU時間(ms) | GPU時間(ms) | 高速化率 |
|--------------|------------|------------|----------|
| 行列乗算     | 34.75      | 1.72       | 20.17倍  |
| 要素ごとの加算 | 4.48       | 0.17       | 27.12倍  |
| 要素ごとの乗算 | 5.39       | 0.67       | 8.03倍   |
| 平方根       | 3.88       | <0.01      | >100倍   |

行列サイズが大きくなるほど、高速化の効果が顕著になります。

## 高度な使用例

### 機械学習の前処理

```python
# 大規模データセットの前処理
X_train = to_gpu(X_train_np)  # トレーニングデータをGPUに転送

# 標準化処理
mean = cuda.mean(X_train, axis=0)
std = cuda.sqrt(cuda.mean(cuda.power(cuda.subtract(X_train, mean), 2), axis=0))
X_train_scaled = cuda.divide(cuda.subtract(X_train, mean), std)

X_train_scaled_np = to_cpu(X_train_scaled)  # 結果をCPUに戻す
```

### メモリ管理

```python
# 明示的なメモリ管理
cuda.clear_memory()  # 不要なGPUメモリを解放

# メモリ使用状況の確認
mem_info = cuda.memory_info()
print(f"GPU使用メモリ: {mem_info['used_gb']:.2f} GB / {mem_info['total_gb']:.2f} GB")
```

## トラブルシューティング

- **ImportError: No module named 'cupy'**: CuPyがインストールされていません。`pip install cupy-cuda11x`（CUDAバージョンに合わせて）でインストールするか、CPUモードで実行してください。
- **メモリエラー**: `cuda.clear_memory()`を呼び出すか、より小さなバッチサイズで処理してください。
- **GPU関連のエラー**: GPUの初期化に問題がある場合は、`SimpleCUDA(force_cpu=True)`でCPUモードに強制できます。
あとはAIに聞いてください


## ライセンス

MIT License

### 免責
SimpleCUDAは「現状のまま」提供されており、明示または黙示を問わず、いかなる種類の保証も行いません。このライブラリは教育・研究・一般的な計算タスク向けに開発されていますが、バグや予期しない動作が含まれている可能性があります。
開発者およびコントリビューターは、このソフトウェアの使用によって生じたいかなる直接的、間接的、偶発的、特殊、典型的、または結果的な損害に対しても責任を負いません。
重要な注意事項: SimpleCUDAは下記のような用途での使用を想定していません。

ミッションクリティカルなシステム
安全性が重要な応用分野
金融取引や重要なビジネスデータの処理
医療機器や人命に関わるシステム
高い信頼性が求められる本番環境

実験的または重要でないプロジェクトでの使用を推奨します。本番環境で使用する前に、徹底的なテストと検証を行ってください。


# SimpleCUDA

## A Lightweight Python Library for GPU Computing with NumPy-like Interface

SimpleCUDA is a simple yet powerful Python library that leverages the computational power of GPUs while maintaining NumPy's familiar syntax. It's designed for scientific computing, machine learning preprocessing, data analysis, and any computational task that requires acceleration, allowing you to harness GPU power with minimal code changes.

### Key Features

- **Intuitive API**: Interfaces similar to NumPy, minimizing learning curve
- **High Performance**: Achieves up to 40x speedup compared to CPU operations
- **Compact Design**: Focused on essential functionality for maximum efficiency
- **Automatic Fallback**: Seamlessly operates on CPU if GPU is unavailable
- **Robust Error Handling**: Minimizes issues from different development environments
- **Bilingual Documentation**: Comprehensive documentation in both English and Japanese

## Requirements

- Python 3.7 or later
- NumPy
- CuPy (for GPU functionality)
- NVIDIA GPU with CUDA support (for GPU functionality)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/simplecuda.git
```

2. Place `simplecuda.py` in the same directory as your Python code or in a location that's in your Python path.

3. Import the library directly with `import simplecuda`:

```python
from simplecuda import cuda, array, to_gpu, to_cpu
```

## Basic Usage

```python
import numpy as np
from simplecuda import cuda, array, matmul, to_cpu

# Create NumPy arrays
a_np = np.random.rand(1000, 1000).astype(np.float32)
b_np = np.random.rand(1000, 1000).astype(np.float32)

# Convert to GPU arrays
a = array(a_np)  # Automatically transfers to GPU
b = array(b_np)

# Perform matrix multiplication on GPU
c = matmul(a, b)

# Transfer result back to CPU
c_np = to_cpu(c)

print(f"Shape of GPU computation result: {c_np.shape}")
```

## Supported Features

### Array Creation

```python
# Various methods for creating arrays
a = array([1, 2, 3, 4])
b = zeros((3, 4))
c = ones((2, 5))
d = eye(3)  # 3x3 identity matrix
e = random((5, 5))  # Uniform random values between 0-1
```

### Basic Operations

```python
# Basic matrix and array operations
c = matmul(a, b)  # Matrix multiplication
d = dot(a, b)     # Dot product
e = add(a, b)     # Element-wise addition
f = subtract(a, b)  # Element-wise subtraction
g = multiply(a, b)  # Element-wise multiplication
h = divide(a, b)    # Element-wise division
```

### Mathematical Functions

```python
# Unary operations and mathematical functions
a_sqrt = sqrt(a)    # Square root
a_exp = exp(a)      # Exponential function
a_log = log(a)      # Natural logarithm
a_abs = abs(a)      # Absolute value
```

### Aggregation Functions

```python
# Data aggregation
total = sum(a)         # Sum
avg = mean(a)          # Average
max_val = max(a)       # Maximum value
min_val = min(a)       # Minimum value
max_idx = argmax(a)    # Index of maximum value
```

### Shape Manipulation

```python
# Shape manipulation
a_reshaped = reshape(a, (2, 5))    # Reshape
a_t = transpose(a)                 # Transpose
```

### Performance Measurement

```python
# Measuring computation time
with cuda.timer("Matrix Multiplication"):
    c = matmul(a, b)
```

## Benchmark Results

Here are benchmark results for operations on 2000x2000 matrices (using NVIDIA RTX3070):

| Operation              | CPU Time (ms) | GPU Time (ms) | Speedup |
|------------------------|--------------|--------------|---------|
| Matrix Multiplication  | 34.75        | 1.72         | 20.17x  |
| Element-wise Addition  | 4.48         | 0.17         | 27.12x  |
| Element-wise Multiply  | 5.39         | 0.67         | 8.03x   |
| Square Root            | 3.88         | <0.01        | >100x   |

The performance benefit becomes more significant as matrix sizes increase.

## Advanced Usage Examples

### Machine Learning Preprocessing

```python
# Preprocessing large datasets
X_train = to_gpu(X_train_np)  # Transfer training data to GPU

# Standardization
mean = cuda.mean(X_train, axis=0)
std = cuda.sqrt(cuda.mean(cuda.power(cuda.subtract(X_train, mean), 2), axis=0))
X_train_scaled = cuda.divide(cuda.subtract(X_train, mean), std)

X_train_scaled_np = to_cpu(X_train_scaled)  # Transfer result back to CPU
```

### Memory Management

```python
# Explicit memory management
cuda.clear_memory()  # Free unused GPU memory

# Check memory usage
mem_info = cuda.memory_info()
print(f"GPU Memory Usage: {mem_info['used_gb']:.2f} GB / {mem_info['total_gb']:.2f} GB")
```

## Troubleshooting

- **ImportError: No module named 'cupy'**: CuPy is not installed. Install with `pip install cupy-cuda11x` (adjust for your CUDA version) or run in CPU mode.
- **Memory errors**: Call `cuda.clear_memory()` or process in smaller batches.
- **GPU-related errors**: If there are issues initializing the GPU, you can force CPU mode with `SimpleCUDA(force_cpu=True)`.

## License

MIT License

## Disclaimer
SimpleCUDA is provided "as is" without warranty of any kind, either expressed or implied. While this library is developed for educational, research, and general computation tasks, it may contain bugs or unexpected behaviors.
The developers and contributors shall not be held liable for any direct, indirect, incidental, special, exemplary, or consequential damages arising from the use of this software.
Important Note: SimpleCUDA is not intended for use in:

Mission-critical systems
Safety-critical applications
Financial transactions or critical business data processing
Medical devices or life-dependent systems
Production environments requiring high reliability

We recommend using it for experimental or non-critical projects. Before using in any production environment, please conduct thorough testing and validation.

