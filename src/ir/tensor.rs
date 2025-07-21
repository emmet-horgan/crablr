use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;
use onnx_ir::{OnnxGraph, NodeType, ArgType, ElementType};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TensorId(pub Uuid);

impl TensorId {
    fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct NodeId(pub Uuid);

impl NodeId {
    fn new() -> Self {
        Self(Uuid::new_v4())
    }
}


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

impl From<ElementType> for DType {
    fn from(value: ElementType) -> Self {
        match value {
            ElementType::Float32 => Self::F32,
            ElementType::Int32 => Self::I32,
            _ => panic!("Unsupported element type")
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorType {
    pub shape: Vec<usize>,
    pub dtype: DType
}

impl From<onnx_ir::ir::ArgType> for TensorType {
    
}

impl TensorType {
    
    #[inline]
    fn rank(&self) -> usize {
        self.shape.len()
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub id: TensorId,
    pub ty: TensorType,
    pub name: Option<String>,
    pub produce: Option<NodeId>,
    pub consumers: Vec<NodeId>
}

impl Tensor {
    fn from_onnx_arg(value: onnx_ir::ir::Argument) -> Self {
        match value.ty {
            ArgType::Scalar(t) => {
                let dtype: DType = t.into();
                Tensor {
                    id: TensorId::new(),
                    ty: TensorType {
                        shape: Vec::new(),
                        dtype: dtype
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum OpKind {
    Relu,
    Add,
    MatMul
}

impl From<onnx_ir::ir::NodeType> for OpKind {

    fn from(value: onnx_ir::ir::NodeType) -> Self {
        match value {
            NodeType::Relu => OpKind::Relu,
            NodeType::Add => OpKind::Add,
            NodeType::MatMul => OpKind::MatMul,
            _ => panic!("Unsupported node type")
        }
    }
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

        let mut tensors: HashMap<TensorId, Tensor> = HashMap::new();
        let mut nodes: HashMap<NodeId, Node> = HashMap::new();

        let mut tensor_name_to_id: HashMap<String, TensorId> = HashMap::new();

        // Convert tensors
        let mut node_prod_id = None;
        let mut node_cons_id = None;
        for n in &onnx_graph.nodes {
            let node_id = NodeId::new();
            let node_op: OpKind = n.node_type.into();
            let mut inputs = vec![];
            let mut outputs = vec![];
            for arg in n.inputs {
                if let Some(input_id) = tensor_name_to_id.get(&arg.name) {
                    //let tensor = tensors.get(input_id).unwrap().clone();
                    inputs.push(input_id.clone())
                } else {
                    let input_id = TensorId::new();

                }
            }
        }
    }
}

