//! Generates the Triton gRPC client from the repository's protos with protox,
//! a Rust protobuf compiler, so no `protoc` binary is needed.

fn main() -> anyhow::Result<()> {
    let proto_dir = "../../../proto/triton";
    let descriptors = protox::compile([format!("{proto_dir}/grpc_service.proto")], [proto_dir])?;
    tonic_prost_build::configure()
        .build_server(false)
        .compile_fds(descriptors)?;
    println!("cargo:rerun-if-changed={proto_dir}");
    Ok(())
}
