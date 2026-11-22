use burn_backend::ops::{ActivationOps, AttentionModuleOptions, FloatTensorOps, IntTensorOps, ModuleOps};
use burn_backend::{TensorData, try_read_sync};
use nanai_burn_gna::{GnaBackend, GnaDevice, GnaLibrary};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lib = GnaLibrary::load_default()?;
    let device_count = GnaDevice::get_count(&lib)?;
    println!("Available GNA devices: {}", device_count);

    let device_index = 0;
    let version = GnaDevice::get_version(&lib, device_index)?;
    println!(
        "Device {} version: 0x{:x} ({})",
        device_index,
        version.0,
        version.as_str()
    );

    let device = GnaDevice::open(&lib, device_index)?;
    println!("Opened device {}", device.index());

    let buffer = device.allocate_buffer(1024)?;
    println!(
        "Allocated {} bytes at {:p}",
        buffer.len(),
        buffer.as_raw_ptr()
    );

    drop(buffer);
    println!("Freed memory");

    drop(device);
    println!("Closed device {}", device_index);

    // GnaBackend hardware matmul demonstration
    let gna_device = GnaDevice::default();

    let lhs_i = GnaBackend::int_from_data(TensorData::from([[1i32, 2, 3], [4, 5, 6]]), &gna_device);
    let rhs_i =
        GnaBackend::int_from_data(TensorData::from([[7i32, 8], [9, 1], [2, 3]]), &gna_device);
    let out_i = GnaBackend::int_matmul(lhs_i, rhs_i);
    let data_i = try_read_sync(GnaBackend::int_into_data(out_i))
        .expect("sync read")
        .expect("int_into_data");
    println!(
        "GNA int_matmul result: {:?}",
        data_i.as_slice::<i32>().expect("i32 slice")
    );

    let lhs_f = GnaBackend::float_from_data(
        TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &gna_device,
    );
    let rhs_f = GnaBackend::float_from_data(
        TensorData::from([[7.0f32, 8.0], [9.0, 1.0], [2.0, 3.0]]),
        &gna_device,
    );
    let out_f = GnaBackend::float_matmul(lhs_f, rhs_f);
    let data_f = try_read_sync(GnaBackend::float_into_data(out_f))
        .expect("sync read")
        .expect("float_into_data");
    println!(
        "GNA float_matmul result: {:?}",
        data_f.as_slice::<f32>().expect("f32 slice")
    );

    // 2-layer MLP forward pass (Linear -> ReLU -> Linear) on GNA hardware
    let x = GnaBackend::float_from_data(TensorData::from([[1.0f32, -2.0, 0.5]]), &gna_device);
    let w1 = GnaBackend::float_from_data(
        TensorData::from([[1.0f32, -1.0], [0.5, 1.0], [2.0, 0.0]]),
        &gna_device,
    );
    let b1 = GnaBackend::float_from_data(TensorData::from([[0.5f32, 0.0]]), &gna_device);
    let h1 = GnaBackend::relu(GnaBackend::float_add(GnaBackend::float_matmul(x, w1), b1));
    let w2 = GnaBackend::float_from_data(TensorData::from([[2.0f32], [-1.0]]), &gna_device);
    let out_mlp = GnaBackend::float_matmul(h1, w2);
    let mlp_data = try_read_sync(GnaBackend::float_into_data(out_mlp))
        .expect("sync read")
        .expect("float_into_data");
    println!(
        "GNA 2-layer MLP result: {:?}",
        mlp_data.as_slice::<f32>().expect("f32 slice")
    );

    // Tiled FlashAttention on GNA hardware
    let q = GnaBackend::float_from_data(TensorData::from([[[[1.0f32, 0.0], [0.0, 1.0]]]]), &gna_device);
    let k = GnaBackend::float_from_data(TensorData::from([[[[1.0f32, 0.0], [0.0, 1.0]]]]), &gna_device);
    let v = GnaBackend::float_from_data(TensorData::from([[[[10.0f32, 20.0], [30.0, 40.0]]]]), &gna_device);
    let attn = GnaBackend::attention(q, k, v, None, None, AttentionModuleOptions::default());
    let attn_data = try_read_sync(GnaBackend::float_into_data(attn))
        .expect("sync read")
        .expect("float_into_data");
    println!(
        "GNA FlashAttention result: {:?}",
        attn_data.as_slice::<f32>().expect("f32 slice")
    );
    if let Some(stats) = GnaBackend::last_performance_stats() {
        println!(
            "GNA HW stats: total_cycles={}, stall_cycles={}, hw_usage={:.2}%",
            stats.total_cycles,
            stats.stall_cycles,
            stats.hw_usage_percentage()
        );
    }

    Ok(())
}
