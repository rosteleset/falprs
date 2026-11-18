//! Generates the Triton gRPC messages from the repository's protos with protox,
//! a Rust protobuf compiler, so no `protoc` binary is needed.

fn main() -> anyhow::Result<()> {
    let proto_dir = "../../proto/triton";
    let descriptors = protox::compile([format!("{proto_dir}/grpc_service.proto")], [proto_dir])?;
    tonic_prost_build::configure()
        // Only the messages: the requests are framed and sent by hand, see triton.rs
        .build_server(false)
        .build_client(false)
        // Output tensors as slices of the received message, not copies of them
        .bytes(".inference.ModelInferResponse.raw_output_contents")
        .compile_fds(descriptors)?;
    println!("cargo:rerun-if-changed={proto_dir}");
    Ok(())
}
