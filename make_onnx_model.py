# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "onnx",
# ]
# ///


import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

def create_simple_onnx_model(output_path="simple_model.onnx"):
    """
    Creates a simple ONNX model: Y = ReLU(A @ X + B)
    - X: Input tensor (e.g., batch_size x 5)
    - A: Weight matrix (constant, 5 x 10)
    - B: Bias vector (constant, 10)
    - MatMul: A @ X
    - Add: (A @ X) + B
    - Relu: ReLU((A @ X) + B)
    """

    # 1. Define graph inputs
    # Input X: batch_size x 5. Using a symbolic dimension for batch_size.
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 5])

    # 2. Define initializers (constants)
    # Weight matrix A: 5x10
    A_val = np.random.rand(5, 10).astype(np.float32)
    A = helper.make_tensor('A', TensorProto.FLOAT, A_val.shape, A_val.flatten().tolist())

    # Bias vector B: 10
    B_val = np.random.rand(10).astype(np.float32)
    B = helper.make_tensor('B', TensorProto.FLOAT, B_val.shape, B_val.flatten().tolist())

    # 3. Define intermediate and output tensors
    # Output of MatMul: Z1 = A @ X. Shape: batch_size x 10
    Z1 = helper.make_tensor_value_info('Z1', TensorProto.FLOAT, [None, 10])
    # Output of Add: Z2 = Z1 + B. Shape: batch_size x 10
    Z2 = helper.make_tensor_value_info('Z2', TensorProto.FLOAT, [None, 10])
    # Final Output Y: Y = ReLU(Z2). Shape: batch_size x 10
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 10])

    # 4. Define nodes
    # MatMul node: Z1 = X @ A (ONNX MatMul is input1 @ input2)
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['X', 'A'], # Note: ONNX MatMul is typically A @ B, so X @ A here.
        outputs=['Z1'],
        name='matmul_op'
    )

    # Add node: Z2 = Z1 + B
    add_node = helper.make_node(
        'Add',
        inputs=['Z1', 'B'],
        outputs=['Z2'],
        name='add_op'
    )

    # Relu node: Y = ReLU(Z2)
    relu_node = helper.make_node(
        'Relu',
        inputs=['Z2'],
        outputs=['Y'],
        name='relu_op'
    )

    # 5. Create the graph
    graph_def = helper.make_graph(
        [matmul_node, add_node, relu_node], # Nodes in topological order
        'simple-linear-model',             # Graph name
        [X],                               # Graph inputs
        [Y],                               # Graph outputs
        [A, B]                             # Initializers
    )

    # 6. Create the model
    model_def = helper.make_model(graph_def, producer_name='simple-model-generator')

    # 7. Check and save the model
    onnx.checker.check_model(model_def)
    onnx.save(model_def, output_path)
    print(f"Model saved to {output_path}")

    print(helper.printable_graph(model_def.graph))

def read_simple_model():
    model = onnx.load('simple_model.onnx')
    for node in model.graph.node:
        print(node.op_type)

if __name__ == "__main__":
    #create_simple_onnx_model()
    read_simple_model()

