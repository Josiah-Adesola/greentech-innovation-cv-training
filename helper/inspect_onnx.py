import onnx

# Load your exported ONNX model
model = onnx.load("best.onnx")

# Get model graph info
print("✅ Model loaded successfully.")
print("Inputs:")
for inp in model.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f" - {inp.name}: {shape}")

print("\nOutputs:")
for out in model.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f" - {out.name}: {shape}")

# Check last layer for number of output channels
print("\nChecking output tensor details...")
for node in model.graph.node[-5:]:
    print(f"Node: {node.name}, OpType: {node.op_type}")
