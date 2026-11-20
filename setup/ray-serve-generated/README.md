Generated Serve protos for the pinned Ray branch (`serve.proto` at
`1842b598a1`). They are not in Ray's git tree; SETUP.md copies them into the
checkout. Regenerate after `src/ray/protobuf/serve.proto` changes:

```bash
cd /mnt/local_storage/ray && OUT=$(mktemp -d)
python -m grpc_tools.protoc -I. --python_out=$OUT --grpc_python_out=$OUT src/ray/protobuf/serve.proto
cd $OUT/src/ray/protobuf
# Ray imports these as ray.serve.generated.*; without the renames Serve fails
# with "cannot pickle 'google._upb._message.Descriptor' object".
sed -i "s/'src.ray.protobuf.serve_pb2'/'ray.serve.generated.serve_pb2'/" serve_pb2.py
sed -i 's/^from src\.ray\.protobuf import serve_pb2 as/from . import serve_pb2 as/' serve_pb2_grpc.py
cp serve_pb2.py serve_pb2_grpc.py /mnt/local_storage/ray/python/ray/serve/generated/
```
