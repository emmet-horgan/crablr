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
    pub shape: Vec<Option<usize>>,
    pub dtype: DType
}

impl From<onnx_ir::ir::ArgType> for TensorType {
    fn from(value: onnx_ir::ir::ArgType) -> Self {
        match value {
            ArgType::Scalar(t) => {
                Self {
                    shape: Vec::new(),
                    dtype: t.into()
                }
            }, 
            ArgType::Tensor(t) => {
                let shape: Vec<Option<usize>>;
                if let Some(static_shape) = t.static_shape {
                    shape = static_shape
                        .clone()
                        .iter()
                        .map(|x| Some(*x))
                        .collect();
                } else {
                    shape = vec![None; t.rank];
                    //panic!("Only static shapes are supported currently, {:?}", t.rank)
                }
                Self {
                    shape: shape,
                    dtype: t.elem_type.into()
                }
            },
            ArgType::Shape(_) => {
                panic!("Cannot construct a shape tensor currently")
            }
        }
    }
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


#[derive(Clone, Debug)]
pub enum OpKind {
    Relu,
    Add,
    MatMul,
    Linear // MatMul followed by Add
}

impl From<onnx_ir::ir::NodeType> for OpKind {

    fn from(value: onnx_ir::ir::NodeType) -> Self {
        match value {
            NodeType::Relu => OpKind::Relu,
            NodeType::Add => OpKind::Add,
            NodeType::MatMul => OpKind::MatMul,
            NodeType::Linear => OpKind::Linear,
            _ => panic!("Unsupported node type: {:?}", value)
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
        //et mut node_prod_id = None;
        //et mut node_cons_id = None;
        for n in &onnx_graph.nodes {
            let node_id = NodeId::new();
            let node_op: OpKind = n.node_type.clone().into();
            let mut inputs = vec![];
            let mut outputs = vec![];
            for arg in &n.inputs {
                if let Some(input_id) = tensor_name_to_id.get(&arg.name) {
                    inputs.push(input_id.clone());
                    let tensor = tensors.get_mut(&input_id).unwrap();
                    tensor.consumers.push(node_id.clone());
                } else {
                    let input_id = TensorId::new();
                    let tensor = Tensor {
                        id: input_id,
                        ty: arg.ty.clone().into(),
                        name: Some(arg.name.clone()),
                        produce: None,
                        consumers: vec![node_id.clone()]
                    };
                    tensors.insert(input_id.clone(), tensor);
                    tensor_name_to_id.insert(arg.name.clone(), input_id);
                    inputs.push(input_id);
                }
            }

            for arg in &n.outputs {
                if let Some(output_id) = tensor_name_to_id.get(&arg.name) {
                    panic!("Cycle detected");
                    //outputs.push(output_id.clone());
                    //let tensor = tensors.get_mut(&output_id).unwrap();
                    //if let Some(_) = tensor.produce {
                    //    panic!("Tensor {:?} cannot have two producers", tensor)
                    //} else {
                    //    tensor.produce = Some(node_id.clone());
                    //}
                } else {
                    let output_id = TensorId::new();
                    let tensor = Tensor {
                        id: output_id,
                        ty: arg.ty.clone().into(),
                        name: Some(arg.name.clone()),
                        produce: Some(node_id.clone()),
                        consumers: vec![]
                    };
                    tensors.insert(output_id.clone(), tensor);
                    tensor_name_to_id.insert(arg.name.clone(), output_id);
                    outputs.push(output_id);
                }
            }
            let node = Node {
                id: node_id.clone(),
                op: node_op,
                inputs,
                outputs,
                name: Some(n.name.clone())
            };
            nodes.insert(node_id, node);
        }
        let inputs = onnx_graph.inputs.iter()
            .filter_map(|a| tensor_name_to_id.get(&a.name).copied())
            .collect();
        
        let outputs = onnx_graph.outputs.iter()
            .filter_map(|a| tensor_name_to_id.get(&a.name).copied())
            .collect();

        Graph {
            nodes,
            tensors,
            inputs,
            outputs
        }
    }
}

#[cfg(test)]
mod tests {

    use std::{path::PathBuf, str::FromStr};

    use super::*;

    #[test]
    fn test_working() {
        let graph = Graph::from_onnx_file(&PathBuf::from_str("./simple_model.onnx").unwrap());
        println!("{:#?}", graph);
    }
}