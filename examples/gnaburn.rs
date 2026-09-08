use nanai_burn_gna::{GnaDevice, GnaLibrary};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lib = GnaLibrary::load_default()?;
    let device_count = GnaDevice::get_count(&lib)?;
    println!("Available GNA devices: {}", device_count);

    let device_index = 0;
    let version = GnaDevice::get_version(&lib, device_index)?;
    println!("Device {} version: 0x{:x} ({})", device_index, version.0, version.as_str());

    let device = GnaDevice::open(&lib, device_index)?;
    println!("Opened device {}", device.index());

    let buffer = device.allocate_buffer(1024)?;
    println!("Allocated {} bytes at {:p}", buffer.len(), buffer.as_raw_ptr());

    drop(buffer);
    println!("Freed memory");

    drop(device);
    println!("Closed device {}", device_index);

    Ok(())
}

