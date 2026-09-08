# nanai-burn-gna

Intel GNA (Gaussian and Neural Accelerator) の動的ロードインターフェイスを提供する [Burn](https://github.com/tracel-ai/burn) 向けクレートです。

本プロジェクト（`nanai-burn-gna`）および基盤となるラッパーインターフェイス（`nanai-gna-dll-rs`）は **MIT OR Apache-2.0** ライセンスで提供されています。
ランタイム時に Intel GNA の DLL（LGPL 等のライセンスで配布される共有ライブラリ）を動的にロード（Dynamic Loading）して利用する設計となっています。

---

## 概要とアーキテクチャ

Intel GNA を深層学習フレームワーク Burn 等から扱うためのロード層を提供します。

```text
+-------------------------------------------------------------+
|                     User Application / Burn                 |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|             nanai-burn-gna  (MIT OR Apache-2.0)             |
|   - Burn向けGNAロード/デバイスインターフェイス                    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|             nanai-gna-dll-rs (MIT OR Apache-2.0)            |
|   - GNA DLL動的バインディング層 (libloading / dynamic load)    |
+-------------------------------------------------------------+
                              : (実行時動的ロード / dlopen, LoadLibrary)
                              v
+-------------------------------------------------------------+
|               Intel GNA 共有ライブラリ (.dll / .so)           |
|                     (LGPL / Proprietary)                    |
+-------------------------------------------------------------+
```

### ライセンス構造の特徴
- **動的ロード方式**: コンパイル時に GNA の共有ライブラリを静的リンクせず、実行時に動的ロード（`libloading`）します。
- **独立性**: 本クレート自体は MIT または Apache-2.0 ライセンスとして自由に組み込み・再配布が可能です。
- **LGPL準拠性**: LGPL ライセンスが適用される GNA DLL はバイナリに同梱・静的結合されず、ユーザー環境の共有ライブラリを実行時に参照するため、LGPL の要件を侵害することなく MIT/Apache-2.0 ソフトウェアから安全に利用できます。

---

## 必要要件

- Rust (Edition 2024 以降推奨)
- Intel GNA ドライバ / ランタイムライブラリ（`gna.dll` 等）がインストールされている環境、またはシステムパスに通っていること

---

## 使い方

### 1. 依存関係の追加

`Cargo.toml` に以下を追加します：

```toml
[dependencies]
burn = "^0.21"
nanai-burn-gna = { path = "../nanai-burn-gna" } # または git リポジトリ
```

### 2. サンプルコード

GNA ライブラリを動的ロードし、デバイス情報の取得やメモリ確保を行う例です：

```rust
use nanai_burn_gna::{GnaDevice, GnaLibrary};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. GNA DLLの動的ロード
    let lib = GnaLibrary::load_default()?;

    // 2. 利用可能なデバイス数の取得
    let device_count = GnaDevice::get_count(&lib)?;
    println!("Available GNA devices: {}", device_count);

    if device_count > 0 {
        let device_index = 0;
        let version = GnaDevice::get_version(&lib, device_index)?;
        println!("Device {} version: 0x{:x} ({})", device_index, version.0, version.as_str());

        // 3. デバイスのオープン
        let device = GnaDevice::open(&lib, device_index)?;
        println!("Opened device {}", device.index());

        // 4. メモリバッファの確保
        let buffer = device.allocate_buffer(1024)?;
        println!("Allocated {} bytes at {:p}", buffer.len(), buffer.as_raw_ptr());

        // RAIIによりスコープ終了時に自動解放
    }

    Ok(())
}
```

---

## サンプルの実行

```bash
cargo run --example gnaburn
```

---

## ライセンス (License)

本プロジェクト（`nanai-burn-gna`）は、以下のいずれかのライセンスの下で提供されます。

- **Apache License, Version 2.0** ([LICENSE-APACHE](http://www.apache.org/licenses/LICENSE-2.0))
- **MIT License** ([LICENSE-MIT](http://opensource.org/licenses/MIT))

※ 実行時にロードされる Intel GNA ランタイム共有ライブラリ（DLL 等）の著作権およびライセンス条件は、それぞれの提供元（Intel 等）に帰属します。
