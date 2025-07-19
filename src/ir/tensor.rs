use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;
use onnx_ir::{OnnxGraph, NodeType}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TensorId(pub Uuid);

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct NodeId(pub Uuid);


#[derive(Clone, Debug)]
pub enum DType {
    F32,
    I8,
    I16,
    I32,
    U8,
    U16,
    U32
}

#[derive(Clone, Debug)]
pub struct TensorType {
    pub shape: Vec<usize>,
    pub dtype: DType
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub id: TensorId,
    pub ty: TensorType,
    pub name: Option<String>,
    pub produce: Option<NodeId>,
    pub consumers: Vec<NodeId>
}

#[derive(Clone, Debug)]
pub enum OpKind {
    Relu,
    Add,
    MatMul
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub op: OpKind,
    pub inputs: Vec<TensorId>, // Tensor IDs
    pub outputs: Vec<TensorId>, // Tensor IDs
    pub name: Option<String>
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub nodes: HashMap<NodeId, Node>,
    pub tensors: HashMap<TensorId, Tensor>,
    pub inputs: Vec<TensorId>, // Input tensor IDs
    pub outputs: Vec<TensorId> // Output tensor IDs
}

impl Graph {

    pub fn from_onnx_file(path: &Path) -> Self {
        let onnx_graph = onnx_ir::parse_onnx(path);

        let mut tensors = HashMap::new();
        let mut nodes = HashMap::new();

        let mut tensor_name_to_id = HashMap::new();

        // Convert tensors
        let mut node_prod_id = None;
        let mut node_cons_id = None;
        for n in &onnx_graph.nodes {
            let node_id = NodeId(Uuid::new_v4());
            let node_op = match n.node_type {
                &NodeType::Relu => OpKind::Relu,
                &NodeType::Add => OpKind::Add,
                &NodeType::MatMul => OpKind::MatMul,
                _ => panic!("Unsupported node type")
            };
            let mut inputs = vec![];
            let mut outputs = vec![];
            for arg in n.inputs {
                
            }
        }
    }
}

