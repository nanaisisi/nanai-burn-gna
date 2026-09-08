pub use nanai_gna_dll_rs::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_gna_library() {
        let result = GnaLibrary::load_default();
        if let Err(err) = result {
            eprintln!("GNA load default: {err}");
        }
    }
}

